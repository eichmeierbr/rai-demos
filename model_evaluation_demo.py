import streamlit as st





class ModelEvaluationParams:
    def __init__(self):
        self.run_streamlit = True
        self.model_list = ('mitll_cifar_l2_1_0.pt', 'mitll_cifar_nat.pt')
        self.model_name = ''
        self.batch_size = 10
        self.num_batches = 10
        self.epsilons = [0.0, 0.25, 0.5, 1.0, 2.0]
        self.learning_rate = 2.5
        self.steps = 10


class ModelEvaluationDemo:
    def __init__(self):
        self.params = ModelEvaluationParams()


    def get_description(self):
        st.markdown("## Model Evaluation")
        with st.expander("Readme",expanded=True):
            st.markdown("""
        
This is where we're going to implement our demo.
        
        """)
        pass


    def load_parameters(self)-> bool:
        with st.sidebar:

            self.params.load_results = st.radio("Load or compute results", 
                                        ('Load', 'Compute'))
            self.params.load_results = self.params.load_results == 'Load'

            form = st.form("Adversarial Form")
            
            form.text('Robustness Comparison Settings')


            if not self.params.load_results:    
                self.params.batch_size = int(form.text_input('Batch Size', '10'))
                self.params.num_batches = int(form.text_input('Number of Batches', '10'))
                self.params.epsilons = form.multiselect("Epsilons", 
                                                        [0.0, 0.25, 0.5, 1.0, 2.0], 
                                                        [0.0, 0.25, 0.5, 1.0, 2.0])
                self.params.epsilons.sort()
                self.params.learning_rate = float(form.text_input('Learning Rate', '2.5'))
                self.params.steps = int(form.text_input('Steps', '10'))

            return form.form_submit_button("Run Demo")


    def run_demo(self)-> None:
        self.get_description()

        submitted = self.load_parameters()
        
        if submitted:
            evaluate_model(self.params)


def evaluate_model(params: ModelEvaluationParams) -> None:
    pass


if __name__=="__main__":
    params = ModelEvaluationParams()
    params.run_streamlit = False

    evaluate_model(params)