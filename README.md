# What is this?

In a prior year, a team named "Wall Street Bets" (Lana Butorovic, Austen Johnson, Joseph Min, and Ryan Schmid) wrote a web app hosted out of [this repo](https://github.com/rws222/fin377-project-site). They used [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/index.html) to plot the efficient frontier and tangency portfolio, and then developed a short quiz to assess the risk aversion parameter for a quadratic utility maximizing investor. With this parameter, they suggested a utlity maximizing portfolio. 

Sadly, their site is no longer working because Heroku, where they hosted it, stopped free services. So I'm porting their project here to demonstrate the use of [Streamlit](https://streamlit.io) for dashboard development and deployment. I've refactored the code in places and added a small tweak to the code to allow for levered portfolios (by shorting the risk free asset).

[You can see this dashboard in action here!](https://donbowen-portfolio-frontier-streamlit-dashboard-app-yentvd.streamlit.app/)

## How to 

If you want to get this app working on your computer so you can use it, play around with it, or modify it, you need:
1. A working python / Anaconda installation
1. Git 

The, open a terminal and run these commands one at a time:

```sh
# download files
cd <path to your FIN377 folder> # make sure the cd isn't a repo or inside a repo!
git clone git@github.com:donbowen/portfolio-frontier-streamlit-dashboard.git

# move to the new folder
cd portfolio-frontier-streamlit-dashboard 

# set up the packages you need for this app to work
conda env create -f environment.yml
conda activate streamlit-env

# start the app in a browser window
streamlit run app.py

# open any IDE you want to modify app 
spyder  # and when you save the file, the app website will update
```

## Further ideas 

1. Easy for me: Add Github action to run `update_data_cache.py` once a month.
1. Easy for anyone: Modify `update_data_cache.py` to download more assets, including non-ETFs. Just add to the list of tickers. 
1. Exploit Streamlit's cache system to make this app faster: 
    - Change code from after the sidebar until the Max Util section to a function that returns the necessary data structures for plotting (instead of doing the plotting via the `pyopt` package), and decorate this function so it is cached
    - Code after that: Use plotly so that output graph is interactive (e.g. zoom in on the interesting part of the graph)
	- see code below for pointers
	- `cov_mat` has asset names, add these as labels to scatterplot
1. Easy for anyone: The requirements file has no version restrictions. We should set exact versions.
	
```python	
## approach 1 for plotly + streamlit: fig.add_trace()

import plotly.graph_objects as go

fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=df.col2,y=df.col1,mode=‘lines’,name=‘line1’))
fig = fig.add_trace(go.Scatter(x=df.col2,y=ucl_array,mode=‘lines’,name=‘ucl =’+str(ucl)))
fig = fig.update_layout(showlegend=True)

st.plotly_chart(fig)	

## approach 2 for plotly + streamlit: 
## make figs with px, then combine via go.Figure(fig1+fig2)

import plotly.express as px
import plotly.graph_objects as go

df   = px.data.iris()

fig1 = px.line(df, x="sepal_width", y="sepal_length")
fig2 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig3 = go.Figure(data=fig1.data + fig2.data)

st.plotly_chart(fig3)	
```

## Notes

While it seems duplicative to have a `requirements.txt` and a  `environment.yml`, the former is needed by Streamlit and the latter makes setting up a conda environment quickly easy. 
