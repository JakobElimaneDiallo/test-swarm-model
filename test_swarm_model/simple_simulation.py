import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
import datetime
from controllers.simple_robot import SimpleRobot
from simple_environment import SimpleEnvironment
from scipy.stats import beta as beta_dist

def main():
    print("Starting simple robot simulation...")
    
    # Set deterministic seed based on user input for reproducibility
    SEED = 42  # chosen by user
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Using random seed: {SEED}")
    
    # Create measurements directory using absolute path
    measurements_dir = '/Users/jakobdiallo/Desktop/RUG/BSc/Year 3/IP Bsc project /Python/test_swarm_model/measurements'
    os.makedirs(measurements_dir, exist_ok=True)
    print(f"\nCreated measurements directory at: {os.path.abspath(measurements_dir)}")
    print(f"Excel file will be saved as: {os.path.join(os.path.abspath(measurements_dir), 'all_robots_measurements.xlsx')}")
    
    # Create measurements DataFrame
    measurements_df = pd.DataFrame(columns=['robot_id', 'x', 'y', 'sample_value', 'belief', 'timestamp'])
    
    # Create environment
    env = SimpleEnvironment(grid_size=5)
    print(f"Created environment with {env.get_vibrating_ratio():.1f}% vibrating squares")
    
    # Create robots
    num_robots = 3
    robots = []
    global_queue = []  # shared message queue
    for i in range(num_robots):
        x = random.random()
        y = random.random()
        robot = SimpleRobot(x, y, env, robot_id=i, message_queue=global_queue)
        # Speed-up parameters
        robot.tau = 0.05  # observe every 0.05 s
        robot.random_walk_duration = 1.0
        robot.min_swarm_count = 400  # require at least 400 observations
        robots.append(robot)
        print(f"Robot {i}: pos=({robot.x:.2f}, {robot.y:.2f})")
    
    def update_sim(frame):
        # Update environment (make vibrating squares pulse)
        env.update_patches(frame)
        
        # Print measurements directory path
        print(f"\nFrame {frame}: Saving measurements to {os.path.abspath(measurements_dir)}")
        
        # Update robots
        dt = 0.1  # time step for robot updates
        for robot in robots:
            robot.update(dt)  # dt = 0.1
            
            # Get and add measurement if available
            measurement = robot.get_last_measurement()
            if measurement:
                # Use simulation time (frame number * dt) as timestamp
                measurement['timestamp'] = frame * dt  # dt = 0.1
                measurements_df.loc[len(measurements_df)] = measurement
                print(f"Added measurement from robot {robot.robot_id} at time {measurement['timestamp']}: {measurement}")
                print(f"Current DataFrame size: {len(measurements_df)}")
            else:
                print(f"No measurement from robot {robot.robot_id} at frame {frame}")
                print(f"DataFrame size: {len(measurements_df)}")
                
        # Update visualization
        for i, robot in enumerate(robots):
            robot_markers[i].set_data([robot.x], [robot.y])
            print(f"Robot {i}: pos=({robot.x:.2f}, {robot.y:.2f}), state={robot.state.name}")
        
        # Save measurements to Excel every 100 frames
        if frame % 100 == 0 and not measurements_df.empty:
            filepath = os.path.join(measurements_dir, 'all_robots_measurements.xlsx')
            measurements_df.to_excel(filepath, index=False)
            print(f"Saved measurements to {filepath}")
        
        return robot_markers + env.patches
    
    # Setup visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Simple Robot Simulation')
    
    # Add environment visualization
    env.add_to_axes(ax)
    
    # Create robot markers
    robot_markers = []
    colors = ['blue', 'red', 'green']
    for i, robot in enumerate(robots):
        marker, = ax.plot(robot.x, robot.y, 'o', color=colors[i], markersize=10, label=f'Robot {i+1}')
        robot_markers.append(marker)
    
    # Add legend
    ax.legend()
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    plt.ion()
    plt.show(block=False)

    try:
        # Run simulation until any robot has made a decision
        print("\nSimulation loopt totdat één robot een besluit heeft genomen...")
        frame = 0
        while not any(robot.decision_flag != -1 for robot in robots):
            update_sim(frame)
            plt.pause(0.1)  # Houd animatie actief
            frame += 1
        deciding_robots = [r.robot_id for r in robots if r.decision_flag != -1]
        print(f"\nRobot(s) {deciding_robots} hebben een besluit genomen!")
        # Print belief of all robots
        for r in robots:
            belief_conf = beta_dist.cdf(0.5, r.alpha, r.beta)
            belief_mean = r.alpha / (r.alpha + r.beta)
            verdict = ">50% oppervlak vibreert" if r.decision_flag == 0 else "<=50% oppervlak vibreert"
            print(f"Robot {r.robot_id}: belief_conf={belief_conf:.3f}, mean={belief_mean:.3f}, sends={r.sends}, recvs={r.recvs}, verdict={verdict}")
        plt.show()
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        print("\nFinal DataFrame contents:")
        print(measurements_df)
        
        # Save final measurements
        if not measurements_df.empty:
            filepath = os.path.join(measurements_dir, 'all_robots_measurements.xlsx')
            print(f"\nSaving final DataFrame to: {os.path.abspath(filepath)}")
            # Delete existing file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            # Save with overwrite
            try:
                measurements_df.to_excel(filepath, index=False)
                print(f"Final measurements saved successfully to: {filepath}")
            except Exception as e:
                print(f"Error saving Excel file: {e}")
                print(f"DataFrame contents: {measurements_df}")
        print("\nSimulation complete!")
        plt.close('all')

if __name__ == '__main__':
    main()
