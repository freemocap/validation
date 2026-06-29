import plotly.graph_objects as go


def create_indicator(value, title, color_of_text = 'black', margins_dict = dict(l=10, r=10, b=10, t=20), size_dict: dict = None):
    fig = go.Figure(go.Indicator(
        mode="number",
        value=value,
        title={'text': title, 'font': {'color': color_of_text, 'size': 35}}, 
        ))
    fig.update_layout(margin=margins_dict)

    if size_dict is not None:
        fig.update_layout(width=size_dict['width'], height=size_dict['height'])


    return fig
