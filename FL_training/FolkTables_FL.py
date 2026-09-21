import numpy as np
import pandas as pd
import csv
import random
import argparse
from pathlib import Path
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from tensorflow import keras
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt


from acs_preprocessing import preprocess_acs_data
from pfl_from_dataframe import *
from fl_aggregation import create_aggregator, get_available_methods, FedAvgAggregator
from client_selection import create_client_selector, get_available_selection_methods


# Mapping from task name to folktables task object
TASK_OBJECTS = {
    'ACSIncome': ACSIncome,
    'ACSEmployment': ACSEmployment,
    'ACSPublicCoverage': ACSPublicCoverage,
    'ACSMobility': ACSMobility,
    'ACSTravelTime': ACSTravelTime,
}


def load_best_federations(csv_path: str = 'results/best_federations.csv') -> dict:
    """
    Load best federations from CSV file.
    
    Returns:
        dict: Nested dictionary {task: {k: [list of states]}}
    """
    df = pd.read_csv(csv_path)
    federations = {}
    
    for _, row in df.iterrows():
        # Skip empty rows
        if pd.isna(row['k']) or pd.isna(row['task']):
            continue
        task = row['task']
        k = int(row['k'])
        states = row['selected_states'].split(',')
        
        if task not in federations:
            federations[task] = {}
        federations[task][k] = states
    
    return federations


def get_optimal_states(task: str, k: int, federations: dict = None, csv_path: str = 'results/best_federations.csv') -> list:
    """
    Get optimal states for a given task and k.
    
    Args:
        task: Task name (e.g., 'ACSIncome', 'ACSEmployment')
        k: Number of states in the federation
        federations: Pre-loaded federations dict (optional)
        csv_path: Path to best_federations.csv
    
    Returns:
        list: List of state codes
    """
    if federations is None:
        federations = load_best_federations(csv_path)
    
    if task not in federations:
        raise ValueError(f"Task '{task}' not found in best federations. Available: {list(federations.keys())}")
    if k not in federations[task]:
        raise ValueError(f"k={k} not found for task '{task}'. Available k values: {list(federations[task].keys())}")
    
    return federations[task][k]



all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']


SEED=random.randint(0, 123494321)

def train_from_model(model, x, y, epch, client_id) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    #10-Fold cross validation
    #kf = KFold(n_plits=10)
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epch, verbose=1)
    plot_learningCurve(history, epch, client_id)
    model.evaluate(x_test, y_test)
    return model



def update_local_model(agg_model, input_shape) :
    #update the local models from the aggregated one received from server
    local_model = tf.keras.models.clone_model(agg_model)
    local_model.build(input_shape)
    local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision')
                ]
            )

    local_model.set_weights(agg_model.get_weights())
    return local_model


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum([tf.convert_to_tensor(grad_list_tuple[i]) for i in range(len(scaled_weight_list))] , axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def FedAvg(models, n, clients_weights, input_shape) :
    scaled_weights = []

    global_model = OurModel(input_shape, 'zeros')
    for i in range(n) :
        scaled_weights.append(scale_model_weights(models[i].get_weights(), clients_weights[i]))
    avg_weights = sum_scaled_weights(scaled_weights)
    global_model.set_weights([avg_weight_layer.numpy() for avg_weight_layer in avg_weights])
    #global_model.set_weights(avg_weights)
    return global_model


def OurModel(input_shape, init_distrib) :
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer = init_distrib, kernel_initializer= init_distrib)(inputs)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer= init_distrib, kernel_initializer= init_distrib )(x)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer= init_distrib, kernel_initializer= init_distrib )(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='Recall'),
        tf.keras.metrics.Precision(name='Precision')

    ]

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
        #run_eagerly=True
    )
    return model


def get_testdata(datasets, min_records):
    sampled_dfs = []
    sampled_labels = []
    for attributes_df, labels_df in datasets:
        # Ensure indices are aligned
        attributes_df = attributes_df.reset_index(drop=True)
        labels_df = labels_df.reset_index(drop=True)

        # Sample min_records from attributes_df
        sampled_attributes_df = attributes_df.sample(n=min_records, random_state=42)

        # Get corresponding labels using the sampled indices
        sampled_labels_df = labels_df.loc[sampled_attributes_df.index]

        sampled_dfs.append(sampled_attributes_df)
        sampled_labels.append(sampled_labels_df)

    # Concatenate all sampled features and labels
    test_features = pd.concat(sampled_dfs, ignore_index=True)
    test_labels = pd.concat(sampled_labels, ignore_index=True)

    return test_features, test_labels

def EOD_and_MAD(model, x_test, y_test) :
    eval_gr_A = model.evaluate(x_test[x_test['SEX_1.0']==True].to_numpy().astype('float'), y_test[x_test['SEX_1.0']==True].to_numpy().astype('int'), verbose=0) #Male
    eval_gr_B = model.evaluate(x_test[x_test['SEX_2.0']==True].to_numpy().astype('float'), y_test[x_test['SEX_2.0']==True].to_numpy().astype('int'), verbose=0) #Female


    acc_gr_A = eval_gr_A[1]
    acc_gr_B = eval_gr_B[1]

    recall_gr_A = eval_gr_A[2]
    recall_gr_B = eval_gr_B[2]
    return ((recall_gr_A - recall_gr_B), (acc_gr_A - acc_gr_B))

def SPD(model, x_test) :
    return (np.mean(model.predict(x_test[x_test['SEX_1.0']==True].to_numpy().astype('float'))) - np.mean(model.predict(x_test[x_test['SEX_2.0']==True].to_numpy().astype('float'))))


def run_training(task, datasets, epochs, max_iterations, centralized_test, 
                 aggregation_method='fedavg', agg_kwargs=None,
                 client_selection='full', selection_kwargs=None):
    """
    Run federated learning training loop.
    
    Args:
        task: Task name
        datasets: List of (features, labels) for each client
        epochs: Local epochs per round
        max_iterations: Number of FL rounds
        centralized_test: Whether to run centralized baseline
        aggregation_method: Aggregation strategy ('fedavg', 'fedprox', 'fedadam', 'scaffold')
        agg_kwargs: Additional kwargs for aggregator (e.g., mu for fedprox)
        client_selection: Client selection method ('full', 'random', 'ucb', 'threshold', 'power_of_choice')
        selection_kwargs: Additional kwargs for client selector
    """
    #Sample a test dataset from all clients' datasets to eval the aggregated model at each iteration
    test_features, test_labels = get_testdata(datasets, 1000)

    n_iterations = 0
    n_clients = len(datasets)
    #how to initilize kernel and bias weights
    init_distrib = tf.initializers.HeUniform(seed=SEED)
    scores = []
    input_shape = (datasets[0][0].shape[1],)
    
    # Setup aggregator kwargs
    if agg_kwargs is None:
        agg_kwargs = {}
    
    # Add n_clients for SCAFFOLD
    if aggregation_method == 'scaffold':
        agg_kwargs['n_clients'] = n_clients

    #Do we want to compare against a centralized training baseline
    #(yes --> do it and save the metrics at the first row of csv file)?
    if centralized_test:
        centralized_x = pd.concat([dataset[0] for dataset in datasets])
        centralized_y = pd.concat([dataset[1] for dataset in datasets])
        centralized_model = OurModel(input_shape, init_distrib)
        print('Centrally training a model on the union data ...')
        centralized_model.fit(centralized_x, centralized_y, epochs=50, verbose=0)
        print('evaluating the centralized model')
        eval = centralized_model.evaluate(test_features, test_labels)
        print(f"centralized model : {[round(s, 5) for s in eval]}")

        #Free heavy data from memory
        del centralized_x
        del centralized_y
        del centralized_model

    # Create aggregator
    print(f"Using aggregation method: {aggregation_method}")
    aggregator = create_aggregator(
        method=aggregation_method,
        input_shape=input_shape,
        model_fn=OurModel,
        learning_rate=0.0001,
        **agg_kwargs
    )
    
    # Initialize global model
    aggregator.initialize_global_model(init_distrib)
    
    ds_sizes = [datasets[i][0].shape[0] for i in range(len(datasets))]
    client_weights = [ds_sizes[i]/sum(ds_sizes) for i in range(len(datasets))]
    
    # Setup client selector
    if selection_kwargs is None:
        selection_kwargs = {}
    
    client_selector = None
    if client_selection != 'full':
        # Default participation rate of 0.5 if not specified
        participation_rate = selection_kwargs.pop('participation_rate', 0.5)
        
        # FedSampling needs client data sizes
        if client_selection == 'fedsampling':
            selection_kwargs['client_data_sizes'] = ds_sizes
        
        client_selector = create_client_selector(
            method=client_selection,
            n_clients=n_clients,
            participation_rate=participation_rate,
            **selection_kwargs
        )
        print(f"Using client selection: {client_selection} with participation rate {participation_rate}")
    
    while True:
        models = []
        selected_clients_list = []
        client_losses = {}
        n_iterations += 1
        
        # Select clients for this round
        if client_selector is not None:
            selected_clients = client_selector.select_clients()
            print(f"Round {n_iterations}: Selected {len(selected_clients)} clients: {selected_clients}")
        else:
            selected_clients = list(range(n_clients))
        
        for i in selected_clients:
            # Prepare local model using aggregator
            model = aggregator.prepare_local_training(client_id=i)
            
            # Reset indices to ensure alignment
            x_train, x_test, y_train, y_test = train_test_split(
                datasets[i][0], datasets[i][1], test_size=0.25
            )
            
            # Local training
            x_train_np = x_train.to_numpy().astype('float')
            y_train_np = y_train.to_numpy().astype('int')
            
            history = model.fit(
                x_train_np, y_train_np,
                validation_split=0.2, batch_size=32, epochs=epochs, verbose=0
            )
            
            # Post-training hook (for SCAFFOLD control variate updates)
            aggregator.post_local_training(client_id=i, local_model=model,
                                          x_train=x_train_np, y_train=y_train_np)
            
            models.append(model)
            selected_clients_list.append(i)
            
            # Get local loss for client selector update
            local_eval = model.evaluate(x_test.to_numpy().astype('float'), 
                                        y_test.to_numpy().astype('int'), verbose=0)
            client_losses[i] = local_eval[0]  # loss
            print(f"model {i}, round {n_iterations} : local performance : {local_eval}")

        # Compute selected client weights (normalized)
        selected_weights = [client_weights[i] for i in selected_clients_list]
        weight_sum = sum(selected_weights)
        selected_weights = [w / weight_sum for w in selected_weights]
        
        # Aggregate models from selected clients
        Agg_model = aggregator.aggregate(models, selected_weights)

        #Evaluate global model on the union validation dataset
        y_pred = Agg_model.predict(test_features.to_numpy().astype('float'))
        y_true = test_labels.to_numpy().astype('int')
        # Clip predictions to avoid log(0) = -inf in log_loss
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        curr_loss = log_loss(y_true, y_pred_clipped)
        curr_balanced_acc = balanced_accuracy_score(y_true, np.round(y_pred))
        curr_recall = recall_score(y_true, np.round(y_pred))
        curr_precision = precision_score(y_true, np.round(y_pred))
        #Fairness
        curr_eod, curr_mad = EOD_and_MAD(Agg_model, test_features, test_labels)
        curr_spd = SPD(Agg_model, test_features)
        score = [curr_loss, curr_balanced_acc, curr_recall, curr_precision, curr_eod, curr_spd, curr_mad]
        scores.append(score)
        
        # Update client selector with this round's results
        if client_selector is not None:
            client_selector.update(selected_clients_list, client_losses)

        for i in range(len(scores)):
            print(f"iteration {i} : {[round(s, 5) for s in scores[i][:2]]} EOD : {scores[i][4]} SPD : {scores[i][5]}")

        if n_iterations == max_iterations:
            break
    return scores



def run_exp(task, states, epochs=1, max_iterations=50, centralized_test=False,
            aggregation_method='fedavg', agg_kwargs=None,
            client_selection='full', selection_kwargs=None):
    """Run federated learning experiment for a given task and states.
    
    Args:
        task: Task name (ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime)
        states: List of state codes to include in the federation
        epochs: Number of epochs per round
        max_iterations: Maximum number of FL rounds
        centralized_test: Whether to run centralized baseline
        aggregation_method: Aggregation strategy ('fedavg', 'fedprox', 'fedadam', 'scaffold')
        agg_kwargs: Additional kwargs for aggregator
        client_selection: Client selection method ('full', 'random', 'ucb', 'threshold', 'power_of_choice')
        selection_kwargs: Additional kwargs for client selector
    
    Returns:
        List of scores per iteration
    """
    if task not in TASK_OBJECTS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_OBJECTS.keys())}")
    
    task_obj = TASK_OBJECTS[task]
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    datasets = []

    for state in states:
        print('Client simulates '+state+' ACS data')
        acs_data = data_source.get_data(states=[state], download=True)
        features, labels, _ = task_obj.df_to_pandas(acs_data)  # Third value is group
        datasets.append([preprocess_acs_data(features), labels])
        print('dataset size : ', datasets[-1][0].shape[0])

    scores = run_training(
        task, datasets, 
        epochs=epochs, 
        max_iterations=max_iterations, 
        centralized_test=centralized_test,
        aggregation_method=aggregation_method,
        agg_kwargs=agg_kwargs,
        client_selection=client_selection,
        selection_kwargs=selection_kwargs
    )
    return scores


def share_more_than_half(list1, list2):
    """Check if two lists share more than 2/3 of their elements."""
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    set1 = set(list1)
    set2 = set(list2)
    shared = set1.intersection(set2)

    return len(shared) > 2 * len(list1) / 3


def run_comparison_experiment(task, k, federations, output_dir='Convergence2', n_seeds=3, seed_start=0,
                               random_agg='fedavg', agg_kwargs=None, skip_optimal=False, only_optimal=False,
                               client_selection='full', selection_kwargs=None):
    """
    Run comparison experiment between optimal and random federations.
    
    Args:
        task: Task name
        k: Federation size (number of states)
        federations: Pre-loaded federations dict from best_federations.csv
        output_dir: Base output directory
        n_seeds: Number of random seeds to run
        seed_start: Starting seed number
        random_agg: Aggregation method for random federations ('fedavg', 'fedprox', 'fedadam', 'scaffold')
        agg_kwargs: Additional arguments for the aggregator (e.g., {'mu': 0.01} for FedProx)
        skip_optimal: If True, skip running optimal federation (use existing results)
        only_optimal: If True, run ONLY optimal federation (skip random federations)
        client_selection: Client selection method ('full', 'random', 'ucb', 'threshold', 'power_of_choice')
        selection_kwargs: Additional kwargs for client selector
    """
    optimal_states = get_optimal_states(task, k, federations)
    print(f"\nRunning comparison for {task} with k={k}")
    print(f"Optimal states: {optimal_states}")
    if only_optimal:
        print("Mode: ONLY optimal federation (FedAvg)")
    else:
        print(f"Optimal aggregation: fedavg | Random aggregation: {random_agg}")
        if client_selection != 'full':
            print(f"Client selection: {client_selection}")
    
    header = ['loss', 'acc', 'recall', 'precision', 'eod', 'spd', 'mad']
    base_path = Path(output_dir) / task / f'fixed_size_{k}'
    
    # Create output directories - include aggregation method and client selection in folder names
    # Proper capitalization for folder names
    agg_name_map = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'fedadam': 'FedAdam', 'scaffold': 'SCAFFOLD'}
    selection_name_map = {'full': '', 'random': 'RandCS', 'ucb': 'UCB', 'threshold': 'Threshold', 'power_of_choice': 'PoC', 'fedsampling': 'FedSamp'}
    random_agg_name = agg_name_map.get(random_agg, random_agg)
    selection_suffix = selection_name_map.get(client_selection, '')
    
    if selection_suffix:
        random_folder = f'Random_{random_agg_name}_{selection_suffix}'
    else:
        random_folder = f'Random_{random_agg_name}'
    optimal_folder = 'Optimal_FedAvg'
    if not only_optimal:
        (base_path / random_folder).mkdir(parents=True, exist_ok=True)
    if not skip_optimal or only_optimal:
        (base_path / optimal_folder).mkdir(parents=True, exist_ok=True)
    
    scores_optimal = []
    scores_random = []
    
    for seed in range(seed_start, seed_start + n_seeds):
        print(f"\n=== Seed {seed} ===")
        
        # Run random federation if not only_optimal mode
        if not only_optimal:
            # Generate random federation that doesn't overlap too much with optimal
            while True:
                random_states = list(np.random.choice(all_states, len(optimal_states), replace=False))
                if not share_more_than_half(optimal_states, random_states):
                    break
            
            print(f"Random states: {random_states}")
            
            # Log random federation
            with open(base_path / random_folder / 'sampled_states.csv', 'a', buffering=1) as f:
                writer = csv.writer(f)
                writer.writerow([list(random_states)])
            
            # Run random federation with specified aggregation method and client selection
            scores_random.append(run_exp(
                task, random_states, 
                aggregation_method=random_agg, 
                agg_kwargs=agg_kwargs,
                client_selection=client_selection,
                selection_kwargs=selection_kwargs.copy() if selection_kwargs else None
            ))
            df = pd.DataFrame(scores_random[-1], columns=header)
            df.to_csv(base_path / random_folder / f'seed_{seed}.csv', index=False, header=True)
        
        # Run optimal federation with FedAvg (baseline) - skip if already have results (unless only_optimal)
        if not skip_optimal or only_optimal:
            # Log optimal federation
            with open(base_path / optimal_folder / 'sampled_states.csv', 'a', buffering=1) as f:
                writer = csv.writer(f)
                writer.writerow([list(optimal_states)])
            
            # Optimal always uses fedavg with full participation (no client selection)
            scores_optimal.append(run_exp(task, optimal_states, aggregation_method='fedavg'))
            df = pd.DataFrame(scores_optimal[-1], columns=header)
            df.to_csv(base_path / optimal_folder / f'seed_{seed}.csv', index=False, header=True)
    
    return scores_optimal, scores_random


def main():
    from fl_aggregation import get_available_methods
    
    parser = argparse.ArgumentParser(description='Run Federated Learning experiments with FolkTables')
    parser.add_argument('--task', type=str, default=None,
                        help='Task to run (ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime). If not specified, runs all tasks.')
    parser.add_argument('--k', type=int, default=None,
                        help='Federation size. If not specified, runs all available k values for the task.')
    parser.add_argument('--federations-csv', type=str, default='results/best_federations.csv',
                        help='Path to best_federations.csv')
    parser.add_argument('--output-dir', type=str, default='Convergence2',
                        help='Output directory for results')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds to run')
    parser.add_argument('--seed-start', type=int, default=0,
                        help='Starting seed number')
    parser.add_argument('--list-available', action='store_true',
                        help='List available tasks and k values from best_federations.csv')
    
    # Aggregation arguments
    parser.add_argument('--random-agg', type=str, default='fedavg',
                        choices=get_available_methods(),
                        help='Aggregation method for random federations (optimal always uses fedavg)')
    parser.add_argument('--fedprox-mu', type=float, default=0.01,
                        help='Proximal term coefficient for FedProx (default: 0.01)')
    parser.add_argument('--fedadam-server-lr', type=float, default=0.01,
                        help='Server learning rate for FedAdam (default: 0.01)')
    parser.add_argument('--fedadam-beta1', type=float, default=0.9,
                        help='Beta1 for FedAdam server optimizer (default: 0.9)')
    parser.add_argument('--fedadam-beta2', type=float, default=0.99,
                        help='Beta2 for FedAdam server optimizer (default: 0.99)')
    parser.add_argument('--fedadam-tau', type=float, default=1e-3,
                        help='Tau (adaptivity) for FedAdam (default: 1e-3)')
    parser.add_argument('--scaffold-local-lr', type=float, default=0.01,
                        help='Local learning rate for SCAFFOLD (uses SGD, default: 0.01)')
    parser.add_argument('--skip-optimal', action='store_true',
                        help='Skip running optimal federation (use existing baseline results)')
    parser.add_argument('--only-optimal', action='store_true',
                        help='Run ONLY optimal federations with FedAvg (skip random federations)')
    
    # Client selection arguments
    parser.add_argument('--client-selection', type=str, default='full',
                        choices=get_available_selection_methods(),
                        help='Client selection method (full=all clients, random, ucb, threshold, power_of_choice, fedsampling)')
    parser.add_argument('--participation-rate', type=float, default=0.5,
                        help='Fraction of clients to select per round (default: 0.5)')
    parser.add_argument('--ucb-exploration', type=float, default=1.0,
                        help='UCB exploration parameter c (default: 1.0)')
    parser.add_argument('--ucb-loss-decay', type=float, default=0.9,
                        help='UCB exponential decay for loss estimates (default: 0.9)')
    parser.add_argument('--threshold-percentile', type=float, default=50.0,
                        help='Threshold percentile for threshold-based selection (default: 50.0)')
    parser.add_argument('--threshold-theta', type=float, default=0.5,
                        help='O-U process mean reversion rate for threshold selector (default: 0.5)')
    parser.add_argument('--poc-d-choices', type=int, default=None,
                        help='Number of candidates for Power of Choice (default: 2 * selected clients)')
    
    args = parser.parse_args()
    
    # Load federations
    federations = load_best_federations(args.federations_csv)
    
    if args.list_available:
        print("Available tasks and k values:")
        for task, k_dict in sorted(federations.items()):
            k_values = sorted(k_dict.keys())
            print(f"  {task}: k = {k_values}")
            for k in k_values:
                print(f"    k={k}: {k_dict[k]}")
        return
    
    # Determine which tasks and k values to run
    if args.task is not None:
        if args.task not in federations:
            print(f"Error: Task '{args.task}' not found. Available: {list(federations.keys())}")
            return
        tasks_to_run = [args.task]
    else:
        tasks_to_run = list(federations.keys())
    
    # Build aggregation kwargs based on method
    agg_kwargs = {}
    if args.random_agg == 'fedprox':
        agg_kwargs['mu'] = args.fedprox_mu
    elif args.random_agg == 'fedadam':
        agg_kwargs['server_lr'] = args.fedadam_server_lr
        agg_kwargs['beta1'] = args.fedadam_beta1
        agg_kwargs['beta2'] = args.fedadam_beta2
        agg_kwargs['tau'] = args.fedadam_tau
    elif args.random_agg == 'scaffold':
        agg_kwargs['local_lr'] = args.scaffold_local_lr
    # fedavg doesn't need extra kwargs
    
    # Build client selection kwargs based on method
    selection_kwargs = {'participation_rate': args.participation_rate}
    if args.client_selection == 'ucb':
        selection_kwargs['exploration_param'] = args.ucb_exploration
        selection_kwargs['loss_decay'] = args.ucb_loss_decay
    elif args.client_selection == 'threshold':
        selection_kwargs['threshold_percentile'] = args.threshold_percentile
        selection_kwargs['theta'] = args.threshold_theta
    elif args.client_selection == 'power_of_choice':
        if args.poc_d_choices is not None:
            selection_kwargs['d_choices'] = args.poc_d_choices
    
    # Run experiments
    for task in tasks_to_run:
        if args.k is not None:
            if args.k not in federations[task]:
                print(f"Warning: k={args.k} not available for {task}. Available: {list(federations[task].keys())}")
                continue
            k_values = [args.k]
        else:
            k_values = list(federations[task].keys())
        
        for k in k_values:
            run_comparison_experiment(
                task=task,
                k=k,
                federations=federations,
                output_dir=args.output_dir,
                n_seeds=args.n_seeds,
                seed_start=args.seed_start,
                random_agg=args.random_agg,
                agg_kwargs=agg_kwargs if agg_kwargs else None,
                skip_optimal=args.skip_optimal,
                only_optimal=args.only_optimal,
                client_selection=args.client_selection,
                selection_kwargs=selection_kwargs if args.client_selection != 'full' else None
            )


if __name__ == '__main__':
    main()
