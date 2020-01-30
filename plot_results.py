import os
import os.path as osp
import numpy as np
import plotly.graph_objects as go

def plot_rewards(data, steps=20):
    layout = go.Layout(
        template='plotly_white',
        xaxis=dict(
            title='Time steps',
            range=[0, 100 * steps],
            showgrid=True,
            showline=False,
            showticklabels=True,
            ticks='outside',
            zeroline=False
        ),
        yaxis=dict(
            title='Average, minimal and maximal reward per episode',
            showgrid=True,
            showline=False,
            showticklabels=True,
            ticks='outside',
            zeroline=False
        ),
    )

    fig = go.Figure(layout=layout)
    for i in range(len(data)):
        x = np.arange(data[i].shape[0]) * steps
        x_rev = x[::-1]
        y = data[i][:, 0]
        y_higher= data[i][:, 1]
        y_lower = data[i][:, 2]
        y_lower = y_lower[::-1]
        fig.add_trace(go.Scatter(
            x=np.hstack([x, x_rev]),
            y=np.hstack([y_higher, y_lower]),
            fill='toself',
            fillcolor=fillcolors[i],
            line_color='rgba(255,255,255,0)',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y,
            line_color=colors[i],
            name=names[i],
        ))
    fig.update_traces(mode='lines')
    fig_name = '{0}.png'.format(algs[i])
    fig.write_image(osp.join(fig_dir, fig_name))
    #fig.show()

if __name__ == '__main__':

    fig_dir = 'figs'
    result_dir = 'results/MARAOrient-v0'
    env_dirs = [osp.join(result_dir, o) for o in os.listdir(result_dir) if osp.isdir(osp.join(result_dir, o))]

    data_ids = [13, 12, 14]
    header_skip = 2

    colors = ['rgb(0,176,246)', 'rgb(231,107,243)']
    fillcolors = ['rgba(0,176,246,0.1)', 'rgba(231,107,243,0.1)']

    algs = ['ppo2_mlp_64', 'ppo2_lstm_64']
    names = ['PPO with MLP2x64', 'PPO2 with MLP2x64']

    data = []
    for i, alg in enumerate(algs):
        ddir = osp.join(result_dir, alg)
        fpath = osp.join(ddir, 'progress.csv')
        try:
            tmp_data = np.genfromtxt(fpath, skip_header=header_skip, delimiter=',')
            mmm = tmp_data[:, np.array(data_ids)]
            mmm = mmm[:, :]
            data.append(mmm)
        except Exception as e:
            print(e)

    plot_rewards(data)



