import streamlit as st

from adversarial_examples_demo import AdversarialPerturbDemo
from concept_probing_demo import ContentProbingDemo
from robustness_comparison_demo import RobustnessComparisonDemo


if __name__ == "__main__":
    st.title('Responsible AI Toolbox Demo')

    st.sidebar.title('Demo Settings')

    demos = {
        'Robustness Comparison': RobustnessComparisonDemo,
        'Adversarial Perturbation': AdversarialPerturbDemo,
        'Concept Probing': ContentProbingDemo,
    }

    demo_mode = st.sidebar.selectbox('Demonstration Mode', demos)
    demo = demos[demo_mode]()

    demo.run_demo()