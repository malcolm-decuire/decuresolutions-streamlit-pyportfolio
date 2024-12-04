#s1 import dependencies
import streamlit as st
import pandas as pd
import plotly.express as px
import time


#s1 set up the page 
st.set_page_config(page_title="ABOUT", page_icon="❓", layout="wide")
st.title("Project Context")
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

#s1 user info 
st.title("❓ About Author")
expander = st.expander("Malcolm Decuire II- Sales Engineer")
expander.write('''
### Lessons via Real-World Projects 
- I was inpsired by the PyPortfolioOpt project because I wanted to blend my hobbies of python development & finance
- My journey into fintech has been deeply inspired by a passion for blending quantitative disciplines—finance, statistics, and computer science—into practical, impactful projects.
- I discovered early in my career that the best way to master the complexities of fintech is by solving real-world problems.

### Lessons for Self-Taught Fintech Developers 
- While exploring topics such as portfolio optimization and risk management, I delved into a wide array of technical resources and academic literature. However, I often found that existing tools fell short in accessibility or adaptability. This gap ignited my drive to create solutions that were not only technically robust but also user-friendly and widely applicable.
- This mindset shaped my approach to fintech innovation: combining theory with action to bridge knowledge gaps and deliver solutions that empower users. It’s this intersection of learning, building, and delivering value that continues to fuel my enthusiasm for the ever-evolving world of financial technology.

''')
 
st.divider()
st.title("❓ About Real Estate Investment Trusts")
expander = st.expander("Recent Trends")
expander.write('''
    ### 1. Increased Adoption of Technology:
    - REITs have increasingly integrated technology to enhance operational efficiency and improve property management.
    - Innovations such as property management software, data analytics, and AI-driven decision-making tools are becoming common.
    - Technology is being leveraged to optimize tenant experiences and streamline property maintenance.
    
    ### 2. Focus on Sustainability and ESG Criteria:
    - There is a growing emphasis on sustainability and Environmental, Social, and Governance (ESG) criteria in the REIT sector.
    - Investors and stakeholders are prioritizing green building practices, energy-efficient upgrades, and corporate responsibility.
    - REITs are adopting sustainable practices to attract environmentally-conscious investors and meet regulatory requirements.
    
    ### 3. Shift Towards Diversified Asset Classes:
    - REITs are diversifying their portfolios beyond traditional office and retail spaces to include sectors like logistics, healthcare, and data centers.
    - This diversification helps mitigate risks and capitalize on emerging market trends.
    - The pandemic has accelerated interest in sectors such as industrial and residential real estate.
    
    ### 4. Increased Focus on Remote Work and Flexible Spaces:
    - The rise of remote work has influenced REITs to adapt their properties to support flexible working arrangements.
    - There is a growing demand for co-working spaces and adaptable office environments.
    - REITs are investing in properties that cater to hybrid work models and offer flexible leasing options.
    ''')
st.divider()

st.title("❓ About Adtech")
expander = st.expander("Recent Trends")
expander.write('''
    ### 1. Rise of Programmatic Advertising:
    - Programmatic advertising like DOOH & instoreOOH, allowing for real-time bidding and automated ad placements.
    - Advertisers are leveraging data to target specific audiences more precisely and optimize ad spend.
    - Advances in AI and machine learning are enhancing the efficiency of programmatic ad systems for custom offerings.

    ### 2. Growth of Privacy-First Marketing:
    - With increasing privacy regulations and concerns, adtech is shifting towards privacy-first approaches.
    - Technologies such as cookieless tracking and consent management platforms are gaining traction.
    - Marketers are adopting strategies that comply with regulations while maintaining effective targeting.

    ### 3. Expansion of Omnichannel Advertising:
    - Omnichannel advertising strategies are becoming more important as brands aim to create seamless customer experiences across various touchpoints.
    - Integration of online and offline data is enabling more cohesive and personalized advertising campaigns.
    - Adtech solutions are evolving to support cross-channel measurement and attribution.

    ### 4. Emergence of AI-Driven Creative Optimization:
    - AI and machine learning are being used to optimize ad creatives in real-time based on performance data.
    - Automated creative testing and dynamic content generation are becoming standard practices.
    - These technologies help in improving ad relevance and engagement by tailoring creatives to individual preferences.
    ''')


#s2b reload 
st.button("🔄 Reload")