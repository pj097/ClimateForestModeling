import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PlotUtils:
    def line_heatmap(self, df, heatmap_cols, row_width):
        line_cols = ['recall', 'precision', 'weightedf1score']
        line_cols += ['prc', 'auc']
        
        line_df = df[[f'val_{c}' for c in line_cols]]
        
        line_df.columns = line_cols
        line_fig = px.line(
            line_df, y=line_cols, 
            line_dash='variable',
        )
        
        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0,
            row_width=row_width
        ) 
        
        for trace in line_fig['data']:
            fig.append_trace(trace, row=1, col=1)
        
        fig.append_trace(
            go.Heatmap(
                z=df[heatmap_cols].T.to_numpy(),
                y=heatmap_cols,
                colorscale=['lavender', 'cornflowerblue', 'tomato'],
                xgap=2, ygap=2, 
                showscale=False,
            ), row=2, col=1
        )
                
        fig.update_xaxes(
            range=(line_df.index.min()-0.5, line_df.index.max()+0.5),
            tickmode = 'linear',
            tick0 = 0.5,
            dtick = 1,
            zeroline=False,
            showticklabels=False,
            title=None,
        ).update_layout(
            legend_title_text=None,
            legend=dict(
                orientation='h',
                xanchor='right', yanchor='bottom',
                x=1, y=1.02   
            ),
            height=500, width=700,
        )
        return fig