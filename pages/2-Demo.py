import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf
from update_data_cache import get_data
from pypfopt.efficient_frontier import EfficientFrontier

pio.renderers.default = 'browser'  # use when doing dev in Spyder (to show figs)

#s1 set up the page 
st.set_page_config(page_title="DEMO", page_icon="▶️", layout="wide")
st.title("PyPortfolioOpt Demo")
st.markdown("##")


#s1a - progress bar
loading_page = "Please Wait 🤲🏽"
progress_text =loading_page
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Python & Finance","Efficient-Frontier Analysis", "Portfolio Allocation"])

# Tab 1: Background
with tab1:
    st.title("❓ Why PyPortfolioOpt")
    expander3 = st.expander
    st.write("Python + Finance/Quantitative Analysis")
    with st.expander("Fintech Development Benefits"):
        # Portfolio Diversification
        st.write("💭 Context: Assume a team of folks w/ finance & adtech experience wanted to trade")
        st.write("👏 That team could leverage tools like Yahoo Finance, EODHD, and PyPortfolioOpt for analysis")
        st.image("pages/photos/yf.png", caption="Yahoo Finance in Action")
        st.image("pages/photos/pyport_2.png", caption="Python in Action")
        st.image("pages/photos/eodhd.jpeg", caption="Fundamental Data in Action")
        st.header("Portfolio Diversification")
        st.write("**Reason:** Tickers like PLD (Prologis, real estate) and IAS (Integral Ad Science, advertising tech) belong to different sectors.")
        st.write("**PyPortfolioOpt Benefit:** Calculates optimal weights for each ticker to maximize diversification and minimize risk.")

        # Efficient Frontier Analysis
        st.header("Efficient Frontier Analysis")
        st.write("**Reason:** Identifying combinations of assets that yield the highest return for a given level of risk is crucial for decision-making.")
        st.write("**PyPortfolioOpt Benefit:** Generates the efficient frontier, showing the best risk-return trade-offs for these tickers.")

        # Risk Assessment and Adjustments
        st.header("Risk Assessment and Adjustments")
        st.write("**Reason:** Tickers have varying volatilities and correlations. For example, real estate tickers (PLD, AMT) might have different risks than tech-focused tickers (IAS, ZETA).")
        st.write("**PyPortfolioOpt Benefit:** Implements advanced risk models, such as shrinkage and exponentially-weighted covariance, to handle noisy financial data.")

        # Data Integration and Alpha Combination
        st.header("Data Integration and Alpha Combination")
        st.write("**Reason:** Financial Data Engineers often aggregate data from fundamental analysis, historical performance, and machine learning models.")
        st.write("**PyPortfolioOpt Benefit:** Combines multiple sources of 'alpha' (e.g., predicted returns) in a risk-efficient way to generate optimal portfolios.")

        # Advanced Features for Complex Portfolios
        st.header("Advanced Features for Complex Portfolios")
        st.write("**Reason:** For tickers like MCHX (Marchex) and ARE (Alexandria Real Estate), traditional methods may not capture market conditions accurately.")
        st.write("**PyPortfolioOpt Benefit:** Experimental features like exponentially-weighted covariance matrices account for changing market dynamics.")

        # Compliance with Investment Rules
        st.header("Compliance with Investment Rules")
        st.write("**Reason:** You are under strict orders to invest in the listed tickers and must make the best use of available data.")
        st.write("**PyPortfolioOpt Benefit:** Ensures that optimization respects constraints, such as including all specified tickers and adhering to risk limits.")


# Tab 2: Portfolio Analysis
with tab2:
    #############################################
    # start: popover for risk assessment
    #############################################
    st.write("PART I: PyPortfolioOpt's mean-variance capabilities")
    toggle_on = st.toggle("Key Terms")
    if toggle_on: 
            st.write("### Tangency Portfolio = The All-Star Team")
            st.write("""
            Imagine you're managing a football team, and you want to win the championship.
            You want players who can score goals (high returns) but also players who defend well and don’t let the other team score (low risk).
            The **Tangency Portfolio** is like the **All-Star Team**: it’s made up of the best mix of attackers, midfielders, and defenders to give you the **highest chance of winning for every level of risk** you’re willing to take.
            In investing, this is like saying, 'I want the best balance between risk and reward.'
            """)

            st.write("### Max Utility Portfolio = Your Dream XI")
            st.write("""
            Now, let’s say you have a unique playing style or strategy as a manager.
            - Maybe you love aggressive, attacking football and want lots of forwards (you’re okay taking more risk for higher rewards).
            - Or maybe you prefer a solid, defensive game where you avoid losing at all costs (you want to minimize risk).
            The **Max Utility Portfolio** is your **Dream XI**: it’s tailored to **your preferences**, focusing on what’s most important to you, even if it’s not the 'statistically best' lineup.
            It’s the team you’d choose based on how you want to play, even if another manager would pick differently.
            """)

            st.write("### Football & Investing")
            st.write("""
            - **Tangency Portfolio**: 'The team with the best stats.' Balances scoring and defending perfectly.
            - **Max Utility Portfolio**: 'Your personalized team.' Matches your playing style, whether that’s attacking, defensive, or balanced.

            In short:
            - The **Tangency Portfolio** is like picking the **perfect balance of stars** who work well together to maximize performance.
            - The **Max Utility Portfolio** is like building **your favorite team**, custom-tailored to your coaching philosophy and what you think will work best!
            """)

    
    with st.expander("Risk-Managment Quiz"):
        qs = {1: [.50, 0, .50, 10],
              2: [.50, 0, .50, 1000],
              3: [.90, 0, .10, 10],
              4: [.90, 0, .10, 1000],
              5: [.25, 0, .75, 100],
              6: [.75, 0, .25, 100]}

        """
        ## Risk Aversion Assessment

        ### Part 1: How much would you pay to enter the following lotteries? (a low number # -> user wants less risky-options)
        """
        ans = {}
        for i in range(1, len(qs) + 1):
            rn = qs[i][0] * qs[i][1] + qs[i][2] * qs[i][3]
            ans[i] = st.slider(
                f'{int(qs[i][0] * 100)}% chance of \${qs[i][1]} and {int(qs[i][2] * 100)}% chance of \${qs[i][3]}',
                0.0, rn, rn, 0.1, key=i)

        risk_aversion = 0
        for i in range(1, len(qs) + 1):
            exp = qs[i][0] * qs[i][1] + qs[i][2] * qs[i][3]
            var = qs[i][0] * (qs[i][1] - exp)**2 + qs[i][2] * (qs[i][3] - exp)**2
            implied_a = 2 * (exp - ans[i]) / var
            risk_aversion += implied_a

        if risk_aversion < 0.000001:  # avoid float error
            risk_aversion = 0.000001

        f'''
        #### Result: Using the survey, your risk aversion parameter is {risk_aversion:.3f}
        ---
        ### If you want, you can override that parameter here: *(limit: 10.00)
        '''
        risk_aversion = st.number_input('Risk Aversion Parameter', 0.000001, float(9), format='%0.2f', value=risk_aversion)

        '''
        ---
        ### Part 2: What is the most leverage/debt-limit are you willing to take on? (limit: 10)
        #### avg. leverage could be between 1.1x to 1.9x depending*
        '''
        leverage = st.number_input('Maximum Leverage', 1, 10, value=1)

    #############################################
    # end: popover for risk assessment
    #############################################

    def get_ef_points(ef, ef_param, ef_param_range):
        mus, sigmas = [], []
        for param_value in ef_param_range:
            try:
                if ef_param == "utility":
                    ef.max_quadratic_utility(param_value)
                elif ef_param == "risk":
                    ef.efficient_risk(param_value)
                elif ef_param == "return":
                    ef.efficient_return(param_value)
                else:
                    raise NotImplementedError("Invalid ef_param")
            except Exception:
                continue
            ret, sigma, _ = ef.portfolio_performance()
            mus.append(ret)
            sigmas.append(sigma)
        return mus, sigmas

    @st.cache_data
    def get_plotting_structures():
        # Always read assets from assets.csv
        asset_file = 'inputs/assets.csv'
        assets_df = pd.read_csv(asset_file, header=None, names=['asset'])
        asset_list = assets_df['asset'].to_list()[:100]  # Max 100 tickers

        # Fetch data dynamically
        e_returns, cov_mat, rf_rate = get_data(asset_list)

        assets = [e_returns, np.sqrt(np.diag(cov_mat))]
        ef = EfficientFrontier(e_returns, cov_mat)
        ef_max_sharpe = EfficientFrontier(e_returns, cov_mat)
        ef_min_vol = EfficientFrontier(e_returns, cov_mat)

        ef_max_sharpe.max_sharpe(risk_free_rate=rf_rate)
        ret_tangent, vol_tangent, sharpe_tangent = ef_max_sharpe.portfolio_performance(risk_free_rate=rf_rate)
        tangency_port = [ret_tangent, vol_tangent, sharpe_tangent]

        ef_min_vol.min_volatility()
        ret_min_vol, vol_min_vol, _ = ef_min_vol.portfolio_performance()

        risk_range = np.logspace(np.log(vol_min_vol + .000001), np.log(assets[1].max()), 20, base=np.e)
        ret_ef, vol_ef = get_ef_points(ef, 'risk', risk_range)
        ef_points = [ret_ef, vol_ef]

        return rf_rate, assets, ef_points, tangency_port

    rf_rate, assets, ef_points, tangency_port = get_plotting_structures()

    mu_cml = np.array([rf_rate, tangency_port[0]])
    cov_cml = np.array([[0, 0], [0, tangency_port[1]]])
    ef_max_util = EfficientFrontier(mu_cml, cov_cml, (-leverage + 1, leverage))

    ef_max_util.max_quadratic_utility(risk_aversion=risk_aversion)
    tang_weight_util_max = ef_max_util.weights[1]
    x_util_max = tang_weight_util_max * tangency_port[1]
    max_util_port = [x_util_max * tangency_port[2] + rf_rate, x_util_max]

    x_high = assets[1].max() * .8
###########################################
## PLOT DESIGN   ##########################
###########################################
# Determine axis ranges to ensure all points are visible
x_min = min(0, min(assets[1]), max_util_port[1], tangency_port[1]) * 0.9  # Add a margin
x_max = max(assets[1].max(), max_util_port[1], tangency_port[1]) * 1.1
y_min = min(0, min(assets[0]), max_util_port[0], tangency_port[0]) * 0.9  # Add a margin
y_max = max(assets[0].max(), max_util_port[0], tangency_port[0]) * 1.1

# Adjust x_min to shift the starting view off-center
x_min_offset = x_min - (x_max - x_min) * 0.05  # Shift slightly left for padding

# Tangency Line (CML)
fig1 = go.Scatter(
    x=[0, x_high],
    y=[rf_rate, rf_rate + x_high * tangency_port[2]],
    mode='lines',
    name='Tangency Line (CML)',  # Added to the legend
    line=dict(color='blue')
)

# Efficient Frontier
fig2 = go.Scatter(
    x=ef_points[1],
    y=ef_points[0],
    mode='lines',
    name='Efficient Frontier',
    line=dict(color='green')
)

# Asset Scatter Points
 # Fetch plotting structures
rf_rate, assets, ef_points, tangency_port = get_plotting_structures()

# Define asset_list explicitly if needed for fig3
asset_list = pd.read_csv('inputs/assets.csv', header=None, names=['asset'])['asset'].to_list()[:100]
   
fig3 = go.Scatter(
    x=assets[1],
    y=assets[0],
    mode='markers',  # Remove text from the markers
    name='Assets',
    marker=dict(size=6, color='orange'),
    customdata=list(zip(asset_list, assets[1], assets[0])),  # Attach asset names and values
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"  # Asset name
        "Volatility (σ): %{customdata[1]:.4f}<br>"  # Rounded x value
        "Expected Return (μ): %{customdata[2]:.4f}"  # Rounded y value
    )
)
# Key Points: Max Utility Portfolio and Tangency Portfolio
fig4 = go.Scatter(
    x=[max_util_port[1], tangency_port[1]],
    y=[max_util_port[0], tangency_port[0]],
    mode='markers+text',
    name='Key Points',
    text=['Max Utility Portfolio', 'Tangency Portfolio'],
    textposition=['middle right', 'top right'],  # Adjusted text positions to avoid overlap
    marker=dict(
        size=12,
        symbol='star',
        color=['purple', 'red']  # Purple for max utility, red for tangency
    ),
    # Text font styling for 'Max Utility Portfolio' and 'Tangency Portfolio'
    textfont=dict(
        color=['#4b0082', 'red'],  # Deep purple for max utility, red for tangency
        size=14,  # Optional: adjust font size for clarity
        family="Arial"  # Optional: adjust font family
    )
)

# Combine all traces into one figure
fig5 = go.Figure(data=[fig1, fig2, fig3, fig4])

# Add axis labels, title, and set axis ranges
fig5.update_layout(
    title="REIT & Adtech Portfolio Analysis",
    xaxis=dict(
        title="Volatility (σ)",
        range=[x_min_offset, x_max]  # Adjusted x-axis range for padding
    ),
    yaxis=dict(
        title="Expected Return (μ)",
        range=[y_min, y_max]  # Set y-axis range
    ),
    legend_title="Legend",
    legend=dict(
        orientation="h",  # Horizontal legend at the top
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Display the updated chart
st.plotly_chart(fig5, use_container_width=True)


#########################################
# Tab 2: Future Expansion (Blank for now)
#########################################    
with tab3:
    st.write("PART II: PyPortfolioOpt's mean-variance capabilities")
    from pypfopt import DiscreteAllocation
    
    #use yf to get of REIT & ADTECH tickers 
    tickers = [
    "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "CCI", "DLR", "O", "CSGP",
    "VICI", "EXR", "AVB", "SBAC", "CBRE", "EQR", "INVH", "VTR", "ARE",
    "RKT", "SUI", "MAA", "AMH", "ESS", "ELS", "GLPI", "WPC", "HST", "UDR", "REG",
    "CPT", "PEAK", "CUBE", "BXP", "OHI", "EGP", "OMC", "LAMR", "IPG", "ZETA",
    "DV", "ZD", "OUT", "MGNI", "STGW", "IAS", "TBLA", "ADV", "PUBM", "QNST",
    "TTGT", "MAX", "CCO", "DSP", "NCMI", "BOC", "ADTH", "QUAD", "CTV", "OB",
    "SCOR", "MCHX", "HHS", "DRCT", "IZEA", "INUV", "MRIN"]

    #download the tickers info
    ohlc = yf.download(tickers, period="max")
    prices = ohlc["Adj Close"].dropna(how="all")
    prices.tail()

    #Calculate Covariance Matrix
    #Long/short min variance 
    from pypfopt import risk_models
    from pypfopt import EfficientFrontier
    # Compute covariance matrix using Ledoit-Wolf shrinkage
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Compute correlation matrix from covariance matrix
    correlation_matrix = pd.DataFrame(S).corr()

    # Create a Plotly heatmap for the correlation matrix
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale="Viridis",  # You can change the color scale if needed
        zmin=-1,  # Correlations range from -1 to 1
        zmax=1
    ))

    # Add titles and layout settings
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Assets",
        yaxis_title="Assets",
        width=800,  # Adjust as needed
        height=800,  # Adjust as needed
    )

    # Expander
    with st.expander("Ledoit-Wolf Method"):
        st.plotly_chart(fig, use_container_width=True)


    #Returns Estimation
    from pypfopt import expected_returns   
    mu = expected_returns.capm_return(prices)
    with st.expander("Expected Returns"):
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Bar(
        y=mu.index,  # Asset names (horizontal axis in barh)
        x=mu.values,  # Corresponding values
        orientation='h',  # Horizontal bar chart
        marker=dict(color='blue')  # Optional: specify color
        ))
        # Customize layout
        fig_returns.update_layout(
            title="Expected Returns by Asset",
            xaxis_title="Expected Return",
            yaxis_title="Assets",
            height=600,  # Adjust the height if needed
            width=800,  # Adjust the width if needed
            )

        # Display the figure in Streamlit
        st.plotly_chart(fig_returns, use_container_width=True)

    # You don't have to provide expected returns in this case
    ef = EfficientFrontier(None, S, weight_bounds=(None, None))
    ef.min_volatility()
    weights = ef.clean_weights()
    weights_series = pd.Series(weights)
    with st.expander("Minimising Variance"):
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(
            y=weights_series.index,  # Asset names
            x=weights_series.values,  # Weights
            orientation='h',
            marker=dict(color='purple')    
        ))
        fig_weights.update_layout(
            title="Global Min. Variance",
            xaxis_title="Weight",
            yaxis_title="Assets",
            height=600,
            width=800,
        )
        # Plot the graph
        st.plotly_chart(fig_weights, use_container_width=True)

    # Clean and prepare the latest prices
    latest_prices = prices.iloc[-1]  # Fetch the last row
    latest_prices = latest_prices.dropna()  # Remove NaN values

    # Ensure `latest_prices` is a pd.Series
    if not isinstance(latest_prices, pd.Series):
        latest_prices = pd.Series(latest_prices)

    # Filter weights to include only assets in latest_prices
    weights = {ticker: weight for ticker, weight in weights.items() if ticker in latest_prices.index}

    # Pass the cleaned `latest_prices` to DiscreteAllocation
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=50000, short_ratio=0.3)
    alloc, leftover = da.greedy_portfolio()

    # Convert allocation to pandas DataFrame for visualization
    alloc_df = pd.DataFrame(list(alloc.items()), columns=["Asset", "Shares"])
    alloc_df["Value"] = alloc_df["Asset"].map(latest_prices) * alloc_df["Shares"]
    alloc_df = alloc_df.sort_values(by="Value", ascending=False)


    # Plot allocation using Plotly as a Pie Chart
    with st.expander("Discrete Allocation"):
        st.write("Given our team started with $50K to")
        fig_allocation = go.Figure()

        # Add a pie chart trace
        fig_allocation.add_trace(go.Pie(
            labels=alloc_df["Asset"],  # Asset names
            values=alloc_df["Value"],  # Allocation values
            hoverinfo='label+percent',  # Show label and percentage on hover
            textinfo='label+percent',  # Show label and percentage on the pie
            texttemplate='%{label}: $%{value:,.2f}K',  # Format values in ($0000.00)
            marker=dict(colors=px.colors.qualitative.Plotly)  # Optional color scheme
        ))

        fig_allocation.update_layout(
            title=f"Portfolio Allocation (Leftover: ${leftover:.2f})",
            height=600,
            width=800,
        )

        # Plot the graph in Streamlit
        st.plotly_chart(fig_allocation, use_container_width=True)


    
    #Max Sharpe w/ Sector Constraints
    sector_mapper = {
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    "EQIX": "Real Estate",
    "WELL": "Real Estate",
    "SPG": "Real Estate",
    "PSA": "Real Estate",
    "CCI": "Real Estate",
    "DLR": "Real Estate",
    "O": "Real Estate",
    "CSGP": "Real Estate",
    "VICI": "Real Estate",
    "0016.HK": "Real Estate",
    "EXR": "Real Estate",
    "AVB": "Real Estate",
    "SBAC": "Real Estate",
    "CBRE": "Real Estate",
    "EQR": "Real Estate",
    "INVH": "Real Estate",
    "VTR": "Real Estate",
    "ARE": "Real Estate",
    "RKT": "Financial Services",
    "SUI": "Real Estate",
    "MAA": "Real Estate",
    "AMH": "Real Estate",
    "ESS": "Real Estate",
    "ELS": "Real Estate",
    "GLPI": "Real Estate",
    "WPC": "Real Estate",
    "HST": "Real Estate",
    "UDR": "Real Estate",
    "REG": "Real Estate",
    "CPT": "Real Estate",
    "PEAK": "Real Estate",
    "CUBE": "Real Estate",
    "BXP": "Real Estate",
    "OHI": "Real Estate",
    "EGP": "Real Estate",
    "OMC": "Media",
    "LAMR": "Media",
    "IPG": "Media",
    "ZETA": "Technology",
    "DV": "Technology",
    "ZD": "Technology",
    "OUT": "Media",
    "MGNI": "Technology",
    "STGW": "Media",
    "IAS": "Technology",
    "TBLA": "Technology",
    "ADV": "Technology",
    "PUBM": "Technology",
    "QNST": "Technology",
    "TTGT": "Technology",
    "MAX": "Technology",
    "CCO": "Media",
    "DSP": "Technology",
    "NCMI": "Media",
    "BOC": "Financial Services",
    "ADTH": "Technology",
    "QUAD": "Media",
    "CTV": "Technology",
    "OB": "Technology",
    "SCOR": "Technology",
    "MCHX": "Technology",
    "HHS": "Media",
    "DRCT": "Technology",
    "IZEA": "Technology",
    "INUV": "Technology",
    "MRIN": "Technology"
    }


    sector_lower = {
        "Media": 0.25,
        "Real Estate": 0.30,
        "Technology": 0.4
    }

    sector_upper = {
        "Media": 0.3,
        "Real Estate": 0.50,
        "Technology": 0.4,
        "Financial Services": 0.1
    } 
    # Step 1: Regularize covariance matrix
    mu = expected_returns.capm_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Step 2: Limit the number of assets (reduces problem size)
    top_assets = mu.nlargest(20).index  # Reduce to top 20 assets
    mu = mu[top_assets]
    S = S.loc[top_assets, top_assets]

    # Step 3: Initialize Efficient Frontier
    ef = EfficientFrontier(mu, S)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    # Step 4: Optimize using a robust solver
    try:
        ef.max_sharpe()  # Use default solver
    except:
        # If OSQP fails, fall back to ECOS
        ef.max_sharpe()
    weights = ef.clean_weights()

    # Step 5: Aggregate weights by sector
    sector_alloc = {}
    for ticker, weight in weights.items():
        if weight > 0:  # Include only non-zero weights
            sector = sector_mapper.get(ticker, "Other")
            sector_alloc[sector] = sector_alloc.get(sector, 0) + weight

    # Step 6: Create sector allocation DataFrame
    sector_alloc_df = pd.DataFrame(list(sector_alloc.items()), columns=["Sector", "Allocation"])

    # Step 7: Display the pie chart
    with st.expander("Sector Constraints"):
        fig_constraints = go.Figure()
        fig_constraints.add_trace(go.Pie(
            labels=sector_alloc_df["Sector"],  # Sector names
            values=sector_alloc_df["Allocation"],  # Allocation values
            hoverinfo='label+percent',
            textinfo='label+percent',
            texttemplate='%{label}: %{value:.2f}',
            marker=dict(colors=px.colors.qualitative.Plotly)
        ))
        fig_constraints.update_layout(
            title="Sector Allocation",
            height=600,
            width=800,
        )
        st.plotly_chart(fig_constraints, use_container_width=True)