import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def analyze_logical_transitions(file_path, session_gap_minutes=30):
    """
    Analyze transitions
    """
    print(f"Analyzing log data from: {file_path}")
    
    # Define our states in the exact order from the structure
    states = [
        'Start', 'PublicContent', 'LoginProcess', 'Overview', 'WatchList', 
        'TradingRelated', 'Account', 'Messages', 'PrivateData', 'Blog', 'Search'
    ]
    
    # Define logical groups - these are the states that require login
    logged_in_states = ['Overview', 'WatchList', 'TradingRelated', 'Account', 'Messages', 'PrivateData']
    public_states = ['PublicContent', 'Blog', 'Search']
    
    # Define logical transitions based on application structure
    logical_flow = {
        'Start': ['PublicContent', 'LoginProcess'],
        'PublicContent': ['LoginProcess', 'Blog', 'Search'],
        'LoginProcess': ['Overview', 'Start'], 
        'Overview': ['PublicContent', 'WatchList', 'TradingRelated', 'Account'],
        'WatchList': ['Overview','PublicContent'],
        'TradingRelated': ['Overview','PublicContent'],
        'Account': ['Overview', 'Messages', 'PrivateData'],
        'Messages': ['Account', 'Overview'],
        'PrivateData': ['Account', 'Overview'],
        'Blog': ['PublicContent', 'Overview'],
        'Search': ['PublicContent', 'Overview'],
    }
    
    # Load data
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    

    
    # Categorize URLs
    df['state'] = df['httpRequest.requestUrl'].apply(categorize_url)
    
    # Identify sessions
    df['login_event'] =  (df['httpRequest.requestUrl'].str.contains('sessions')) & (df['httpRequest.requestMethod'] == 'POST')
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60
    df['new_session'] = df['time_diff'] > session_gap_minutes 
    df['session_id'] = df['new_session'].cumsum()
    
    # Process each session with login awareness
    all_logical_transitions = defaultdict(int)
    
    for session_id, session_df in df.groupby('session_id'):
        # Sort by timestamp
        session_df = session_df.sort_values('timestamp')
        
        # Get sequence of states
        raw_states = session_df['state'].tolist()
        
        # Filter to include only our target states and remove consecutive duplicates
        filtered_states = []
        for state in raw_states:
            if state in states and (not filtered_states or state != filtered_states[-1]):
                filtered_states.append(state)
        
        # Skip sessions with less than 2 states
        if len(filtered_states) < 2:
            continue
        
        # Process the session
        is_logged_in = False
        logical_transitions = []
        current_from_state = None
        # Mark first state as Start if it's not already
        if filtered_states[0] != 'Start':
            logical_transitions.append(('Start', filtered_states[0]))
        
        for i, state in enumerate(filtered_states):
            # Update login state
            if state == 'LoginProcess':
                is_logged_in = True
                if i < len(filtered_states) - 1 and filtered_states[i+1] != 'Overview':
                    # Force Overview after Login
                    logical_transitions.append(('LoginProcess', 'Overview'))
                    current_from_state = 'Overview'
                continue
            
            # Skip if we're at the last state or login handling added a transition
            if i == len(filtered_states) - 1 or current_from_state:
                current_from_state = None
                continue
                
            from_state = state
            to_state = filtered_states[i+1]
            
            # Handle transitions based on login state
            if is_logged_in:
                if from_state in logged_in_states:
                    # Normal logged-in transition
                    if to_state in logical_flow.get(from_state, []):
                        logical_transitions.append((from_state, to_state))
                    else:
                        # Find a logical path
                        if to_state in logged_in_states:
                            # Both states require login, find a logical intermediate
                            if to_state == 'Overview':
                                # Direct to Overview is usually logical
                                logical_transitions.append((from_state, 'Overview'))
                            else:
                                # Go via Overview for most transitions
                                logical_transitions.append((from_state, 'Overview'))
                                logical_transitions.append(('Overview', to_state))
                        else:
                            # Transitioning to public state, go via logout
                            logical_transitions.append((from_state, 'PublicContent'))
                    
                elif from_state in public_states and to_state in logged_in_states:
                    # Going from public to private without login - must be already logged in
                    if to_state == 'Overview':
                        # Direct to Overview is logical
                        logical_transitions.append((from_state, 'Overview'))
                    else:
                        # Other private sections usually go via Overview
                        logical_transitions.append((from_state, 'Overview'))
                        logical_transitions.append(('Overview', to_state))
                
                else:
                    # Normal public transition while logged in
                    logical_transitions.append((from_state, to_state))
            
            else:  # Not logged in
                if to_state in logged_in_states:
                    # Can't access logged-in states without login
                    logical_transitions.append((from_state, 'LoginProcess'))
                    logical_transitions.append(('LoginProcess', 'Overview'))
                    if to_state != 'Overview':
                        logical_transitions.append(('Overview', to_state))
                else:
                    # Normal public transition
                    logical_transitions.append((from_state, to_state))
        
        # Count logical transitions
        for from_state, to_state in logical_transitions:
            all_logical_transitions[(from_state, to_state)] += 1
    
    # Create transition matrix
    matrix = np.zeros((len(states), len(states)))
    
    # Fill with logical transitions
    for i, from_state in enumerate(states):
        total = sum(all_logical_transitions.get((from_state, to_state), 0) for to_state in states)
        
        if total > 0:
            for j, to_state in enumerate(states):
                count = all_logical_transitions.get((from_state, to_state), 0)
                matrix[i, j] = count / total
    
    # Analyze transitions from each state
    for state in states:
        print(f"\nLogical transitions from {state}:")
        transitions = [(k, v) for k, v in all_logical_transitions.items() if k[0] == state]
        transitions.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate total
        total = sum(count for (_, _), count in transitions)
        
        if total > 0:
            for (_, to_state), count in transitions:
                percentage = (count / total) * 100
                print(f"  → {to_state}: {count} ({percentage:.1f}%)")
        else:
            print("  No transitions found")
    
    # Print the matrix
    print("\nLogical Transition Matrix:")
    print("states = [")
    for state in states:
        print(f"    '{state}',")
    print("]")
    print("\ntransition_matrix = np.array([")
    for i, from_state in enumerate(states):
        print(f"    # {from_state}")
        print("    [" + ", ".join(f"{matrix[i, j]:.2f}" for j in range(len(states))) + "],")
    print("])")


    # Print the transition matrix
    
    return matrix, states, all_logical_transitions

def categorize_url(url):
    """Categorize URL into states"""
    if not isinstance(url, str):
        return 'other'
    path = url.lower()
    # categorize based on patterns
    if 'watchlist' in path or 'watchlist_items' in path or 'watchlists' in path:
        return 'WatchList'
    elif 'blogg' in path or 'blog' in path:
        return 'Blog'
    elif 'message' in path or 'unread_status' in path:
        return 'Messages'
    elif ('private' in path or 'price_alarm' in path or 
          'corporate_action' in path or 'kyc' in path or
          'commission' in path):
        return 'PrivateData'
    elif 'search' in path or 'instrument_search' in path or 'main_search' in path:
        return 'Search'
    elif ('instrument' in path or 'trading' in path or 
          'markets' in path):
        return 'TradingRelated'
    elif 'login' in path or 'loggain' in path or 'authentication' in path or 'basic/login' in path or 'nnxapi' in path or 'jwt/refresh' in path:
        return 'LoginProcess'
    elif 'overview' in path or 'oversikt' in path:
        return 'Overview'
    elif 'account' in path:
        return 'Account'
    elif 'public' in path or 'static' in path or 'webmanifest' in path:
        return 'PublicContent'
    
    return 'other'

if __name__ == "__main__":
    try:
        file_path = '../pathTo.csv'  # file path
        matrix, states, transitions = analyze_logical_transitions(file_path)
       # validate_matrix(matrix, states)
        print("\nAnalysis complete - Transition matrix reflects logical application flow.")
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='YlGnBu', interpolation='nearest')
        plt.gca().set_facecolor('white')   # axes background
        plt.gcf().set_facecolor('white')   # figure background
        plt.colorbar()
        plt.xticks(range(len(states)), states, rotation=45, ha='right')
        plt.yticks(range(len(states)), states)
        plt.title('Transition Matrix Heatmap', fontsize=14)
        plt.tight_layout()
        plt.grid(False)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
