""" This is the Code for The Kalman Fliter of The 2013 Helheim GPS data
This is the initiation page"""

from KF_function import Run_Kalman_3D
from Functions.Plotting_function import Plot_all_three

"""For loop that initiates the Kalman filter """
# Nodes_ls = ['Node0001', 'Node0002', 'Node0003', 'Node0004', 'Node0005', 'Node0006',
#             'Node0007', 'Node0008', 'Node0009', 'Node0010', 'Node0011', 'Node0012',
#             'Node0013', 'Node0014', 'Node0015', 'Node0017', 'Node0018', 'Node0019',
#             'Node0020']

Nodes_ls = ["Node0009"]

for node in Nodes_ls:
    print(node)
    sigma_h = 50
    noise_p = 0.00000000000000000005
    noise_v = 0.00000000000000000005
    # Run_Kalman_3D(node, sigma_h, noise_v, noise_p)

    """plot the Kalman Filtered node"""
    node_location = f"Filtered_Nodes/{node}_{noise_v}_{sigma_h}"
    Plot_all_three(node)

