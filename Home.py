
#s1- import dependencies
import streamlit as st
import pandas as pd
import plotly.express as px
import time
# from session_state import SessionState

#s1a set up the page 
st.set_page_config(page_title="HOME", page_icon="💰", layout="wide")
st.header("DECURE LABS FINTECH DEVELOPMENT ")
st.markdown("##")

#s1b progress bar
loading_page = "Please Wait 🤲🏽"
progress_text =loading_page
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

#s1c - info to user remember to leave double spaces after each line '''
st.title("💡 Audience Takeaways")
st.write("📌 Learn about the Adtech & Real-Estate Industry")
st.write("📌 Learn how to deploy a Python app on Snowflake")
st.subheader("Project Summary")
expander2 = st.expander("Fintech Development Inspiration")
expander2.write('''Imagine you're picking a team for a game where some players are fast, some are strong, and some are good at strategy.  ''')
expander2.write('''You want the best team that can win without getting tired too fast.  ''')
expander2.write('''Financial Data Engineers use PyPortfolioOpt like a coach—it helps pick the best mix of players (tickers) so the team is balanced and plays well together ''')
expander2.write('''It’s like making sure you don’t rely too much on one player to win the game!  ''')
st.divider()

#s2 setup
st.header("📝 Visit 'Demo' page")
st.subheader("✏️ Click on each tab to learn about PyPortfolioOpt Library' ")
st.subheader("👉 Click on each PyPortfolioOpt library example")
st.caption("As of Dec-2024: Limited Preview of Demo")
st.subheader("⏭️ Check 'Product Roadmap' page for upcoming features")

st.divider()

st.header("Inspirations")
expander = st.expander("Yahoo Finance & Python ")
expander.write(''' IT IS FREE -> seriously free & effective finance APIs are difficult to find!! ''')
expander.write('''
Yahoo Finance is a leading financial news and data platform that provides real-time stock market updates, financial news, investment insights, and tools for tracking portfolios. 
               It offers detailed information on stocks, currencies, cryptocurrencies, commodities, mutual funds, and ETFs. With a user-friendly interface, 
''')

expander.write('''
Yahoo Finance caters to both individual investors and financial professionals, helping them stay informed with market trends, company reports, and economic developments. 
               It also features advanced charting tools, personal finance tips, and access to global financial markets, making it a versatile resource for financial decision-makin
 ''')
expander.image("pages/photos/yf.png")

expander.write('''
PyPortfolioOpt is a comprehensive library for portfolio optimization, offering classical methods like efficient frontier and Black-Litterman allocation, as well as modern approaches such as shrinkage and Hierarchical Risk Parity.
''')

expander.write('''
Its flexible design caters to both casual investors and professionals, enabling seamless integration with various investment strategies. Whether you're a fundamentals-driven investor or an algorithmic trader, PyPortfolioOpt helps combine alpha sources efficiently while managing risk effectively.
 ''')
expander.image("pages/photos/pyport_2.png")



#s3b create next section on pg
st.divider()

#s4
#placeholder for media 

#s5b reload 
st.button("🔄 Reload")