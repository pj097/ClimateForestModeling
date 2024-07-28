import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PlotUtils:
    def calculate_means_corrs(self, seasons, sentinel_bands):
        corrs = []
        for s in (pbar := tqdm(seasons)):
            pbar.set_description(s)
            corr_path = Path('tmp', f'corr_{s}.csv')
            
            if corr_path.is_file():
                corr = pd.read_csv(corr_path, index_col=[0])
            else:
                shard_means = []
                all_shards = shards_dir.joinpath(f'features_2017{s}').glob('*.npy')
                for f in tqdm(list(all_shards), leave=False):
                    data = np.copy(np.load(f))
                    shard_means.append(data.mean(axis=(0, 1)))
            
                df = pd.DataFrame(shard_means, columns=sentinel_bands.keys())
                corr = df.corr()
                corr.to_csv(corr_path)
                
            corrs.append(corr)
            
        return corrs
        
    def season_correlation(self, seasons, sentinel_bands, animate=True, fig_path=None):
        season_names = ['spring', 'summer', 'autumn', 'winter']
        corrs = self.calculate_means_corrs(seasons, sentinel_bands)
        corrs = [c.round(2) for c in corrs]
        
        if animate:
            fig = px.imshow(
                np.array(corrs),
                animation_frame=0,
                labels=dict(color="Corr coef"),
                x=corrs[0].index,
                y=corrs[0].columns,
                title='Seasonal correlation heatmap',
                text_auto=True, aspect='auto', zmin=0, height=500
            )
            fig.layout.sliders[0]['currentvalue']['prefix'] = ''
            for step in fig.layout.sliders[0].steps:
                step.label = season_names[int(step.label)]
        else:      
            fig = px.imshow(
                np.array(corrs),
                facet_col=0, facet_col_wrap=2,
                x=corrs[0].index,
                y=corrs[0].columns,
                text_auto=True, aspect='auto', zmin=0,
                height=500, width=700,
            )
            
            for ann in fig.layout['annotations']:
                ann['text'] = season_names[int(ann['text'].split('=')[1])]
            if fig_path:
                fig.write_image(fig_path)
        return fig
    
    def zscore(self, X, bands, data_summary):
        stats = {}
        for stat in ['mean', 'std']:
            stats[stat] = np.array(list(data_summary[stat].values()))
            
        normalised_X = (X - stats['mean'][bands])/stats['std'][bands]
        return normalised_X
        
    def feature_hist(self, samples, band_group, all_bands, data_summary, layout_params):
        fig = make_subplots(
            rows=1, cols=2, shared_yaxes='all',
            horizontal_spacing=0.05,
        )
        for band, color in reversed(band_group.items()):
            hist1 = go.Histogram(
                x=samples[band],
                name=band, 
                marker_color=color
            )
            fig.append_trace(hist1, row=1, col=1)
            hist2 = go.Histogram(
                x=self.zscore(
                    samples[band],
                    list(all_bands.keys()).index(band),
                    data_summary,
                ),
                showlegend=False,
                marker_color=color
            )
            fig.append_trace(hist2, row=1, col=2)

            fig.add_vline(
                x=data_summary['mean'][band], 
                line_dash='dash', 
                line_color=color,
                opacity=0.75,
                row=1, col=1,
            )
    
        fig.update_layout(
            layout_params,
            barmode='overlay',
            yaxis_title='Count',
            height=350, width=700,
            legend=dict(
                orientation='h',
                xanchor='right', yanchor='bottom',
                x=1, y=1.02   
            ),
        )
        fig.update_traces(opacity=0.75)
    
        return fig
    
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