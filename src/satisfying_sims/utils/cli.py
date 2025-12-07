import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser(description='Spacenet Training Script')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='experiment_name')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--frame_rate', type=int, default=60, metavar='N',
                        help='frame rate for rendering (default: 60)')
    parser.add_argument('--duration', type=float, default=10.0, metavar='N',
                        help='duration of the simulation in seconds (default: 10.0)')
    parser.add_argument('--outdir', type=str, default='results', metavar='N',
                        help='directory to save the rendered video (default: results)')
    parser.add_argument('--physics_rate_request', type=int, default=600, metavar='N',
                        help='requested physics simulation rate in Hz (default: 600)')
    parser.add_argument('--rules', type=json.loads, default={}, metavar='N',
                        help='list of rules to apply in the simulation (default: empty list)')
    parser.add_argument('--boundary', type=json.loads, default={}, metavar='N',
                        help='type of boundary for the simulation (default: box)')
    parser.add_argument('--n_bodies', type=int, default=10, metavar='N',
                        help='number of bodies in the simulation (default: 10)')
    parser.add_argument('--body_color', type=str, default='blue', metavar='N',
                        help='color of the bodies in the simulation (default: blue)')
    parser.add_argument('--sigma_v', type=float, default=5.0, metavar='N',
                        help='standard deviation of the initial velocity (default: 5.0)')
    parser.add_argument('--radius', type=float, default=1.0, metavar='N',
                        help='radius of the bodies in the simulation (default: 1.0)')
    return parser

'''
usage: python render_video.py --exp_name my_experiment --seed 42 --frame_rate 60 --duration 15.0 \
    --outdir results --physics_rate_request 600 --rules '{"SpawnOnCollision": {}}' \
        --boundary '{"type": "BoxBoundary", "params": {"width": 100.0, "height": 100.0}}' \
            --n_bodies 20 --body_color blue --sigma_v 10.0 --radius 1.0
'''