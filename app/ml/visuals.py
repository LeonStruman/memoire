import streamlit as st


# Menu on the sidebar
def menu():
    with st.sidebar:
        # st.page_link('streamlit_home_page.py', label='Home', icon='🏠')
        st.page_link(
            "pages/visuals_feature_selection.py", label="Feature Selection", icon="🛒"
        )
        st.page_link(
            "pages/visuals_feature_saturation.py",
            label="Feature Selection Saturation",
            icon="📈",
        )
        st.page_link("pages/visuals_experiments.py", label="Experiments", icon="⚙️")
        #st.page_link("pages/visuals_constraints.py", label="Constraints on features", icon="🎛️")


############### Streamlit launch ###############

# Build sidebar menu
menu()

# Title and welcome message
st.title("Visuals for Datagotchi Health")
