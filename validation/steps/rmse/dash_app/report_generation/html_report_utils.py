from plotly.subplots import make_subplots
import plotly.io as pio
from validation.steps.rmse.dash_app.ui_components.indicators import create_indicator

def turn_figs_into_subplot(fig_x, fig_y, fig_z):
    # Create a subplot with 1 column and 3 rows
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1)

    # Add traces for X, Y, and Z
    for i, single_fig in enumerate([fig_x, fig_y, fig_z], start=1):
        for trace in single_fig['data']:
            fig.add_trace(trace, row=i, col=1)

    return fig



def create_position_subplots_from_individual_figs(fig_x, fig_y, fig_z, color_of_cards):

    fig = turn_figs_into_subplot(fig_x, fig_y, fig_z)

    # Update figure layout
    fig.update_layout(height=700, showlegend=True, paper_bgcolor=color_of_cards)
    fig.update_xaxes(title_text="Frame", row=3, col=1)  # Only the last plot (Z) shows the Frame on x-axis
    fig.update_yaxes(title_text="X Axis (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Y Axis (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Z Axis (mm)", row=3, col=1)

    return fig

def create_velocity_subplots_from_individual_figs(fig_x, fig_y, fig_z, color_of_cards):

    fig = turn_figs_into_subplot(fig_x, fig_y, fig_z)

    # Update figure layout
    fig.update_layout(height=700, showlegend=True, paper_bgcolor=color_of_cards)
    fig.update_xaxes(title_text="Frame", row=3, col=1)  # Only the last plot (Z) shows the Frame on x-axis
    fig.update_yaxes(title_text="X Velocity (mm/frame)", row=1, col=1)
    fig.update_yaxes(title_text="Y Velocity (mm/frame)", row=2, col=1)
    fig.update_yaxes(title_text="Z Velocity (mm/frame)", row=3, col=1)

    return fig


def create_position_indicators(rmse_values):
    rmse_total_indicator = create_indicator(rmse_values['total'], "Total Position RMSE")
    rmse_x_indicator = create_indicator(rmse_values['x'], "X Position RMSE", color_of_text='red', margins_dict = dict(l=1, r=1, b=1, t=1))
    rmse_y_indicator = create_indicator(rmse_values['y'], "Y  Position RMSE", color_of_text='green', margins_dict = dict(l=1, r=1, b=1, t=1))
    rmse_z_indicator = create_indicator(rmse_values['z'], "Z Position RMSE", color_of_text='blue', margins_dict = dict(l=1, r=1, b=1, t=1))
    indicators = [rmse_total_indicator, rmse_x_indicator, rmse_y_indicator, rmse_z_indicator]

    return indicators

def create_velocity_indicators(rmse_values):
    rmse_total_indicator = create_indicator(rmse_values['total'], "Total Velocity RMSE")
    rmse_x_indicator = create_indicator(rmse_values['x'], "X Velocity RMSE", color_of_text='red', margins_dict = dict(l=1, r=1, b=1, t=1))
    rmse_y_indicator = create_indicator(rmse_values['y'], "Y  Velocity RMSE", color_of_text='green', margins_dict = dict(l=1, r=1, b=1, t=1))
    rmse_z_indicator = create_indicator(rmse_values['z'], "Z Velocity RMSE", color_of_text='blue', margins_dict = dict(l=1, r=1, b=1, t=1))
    indicators = [rmse_total_indicator, rmse_x_indicator, rmse_y_indicator, rmse_z_indicator]

    return indicators

def fig_to_html(fig):
    return pio.to_html(fig, full_html=False)

def generate_position_rmse_table(rmse_values):
    table_html = '<table style="margin-left:auto; margin-right:auto;">'  # Center the table

    # Add a title row
    table_html += """
    <tr>
        <th colspan='2' style='text-align:center; padding:10px;'>Joint Position RMSE (mm) </th>
    </tr>
    """
    # Add rows for each RMSE value with colored titles
    table_html += f"<tr><td style='color:red;'>X RMSE:</td><td>{rmse_values['x']:.2f}</td></tr>"
    table_html += f"<tr><td style='color:green;'>Y RMSE:</td><td>{rmse_values['y']:.2f}</td></tr>"
    table_html += f"<tr><td style='color:blue;'>Z RMSE:</td><td>{rmse_values['z']:.2f}</td></tr>"
    table_html += '</table>'
    return table_html

def generate_velocity_rmse_table(rmse_values):
    table_html = '<table style="margin-left:auto; margin-right:auto;">'  # Center the table

    # Add a title row
    table_html += """
    <tr>
        <th colspan='2' style='text-align:center; padding:10px;'>Joint Velocity RMSE (mm/frame) </th>
    </tr>
    """
    # Add rows for each RMSE value with colored titles
    table_html += f"<tr><td style='color:red;'>X RMSE:</td><td>{rmse_values['x']:.2f}</td></tr>"
    table_html += f"<tr><td style='color:green;'>Y RMSE:</td><td>{rmse_values['y']:.2f}</td></tr>"
    table_html += f"<tr><td style='color:blue;'>Z RMSE:</td><td>{rmse_values['z']:.2f}</td></tr>"
    table_html += '</table>'
    return table_html