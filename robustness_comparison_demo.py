import streamlit as st

from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision

from rai_toolbox.perturbations import gradient_ascent, AdditivePerturbation
from rai_toolbox.optim import L2ProjectedOptim
from rai_experiments.models.pretrained import load_model

from adversarial_examples_demo import load_CIFAR10



class RobustnessComparisonParams:
    def __init__(self):
        self.load_results = True
        self.run_streamlit = True
        self.model_list = ('mitll_cifar_l2_1_0.pt', 'mitll_cifar_nat.pt')
        self.model_name = ''
        self.batch_size = 10
        self.num_batches = 10
        self.epsilons = [0.0, 0.25, 0.5, 1.0, 2.0]
        self.learning_rate = 2.5
        self.steps = 10


class RobustnessComparisonDemo:
    def __init__(self):
        self.params = RobustnessComparisonParams()


    def get_description(self):
        st.markdown("## Robustness Comparison")
        with st.expander("Readme",expanded=True):
            st.markdown("""
        
This simple demo performs a robustness comparison between a standard model and
a robustly trained model. Both models use the ResNet-50 architecture and perform
classification on the CIFAR-10 dataset. The robust model was trained against 
adversarially-generated training data. The standard model was trained against 
CIFAR-10 using standard (i.e., non-adversarial) data augmentation procedures. 

You can select to load precomputed results, or generate them yourself using the
radio buttons above the settings box in the sidebar. To compute your own results,
select how many examples you want to evaluate (batch_size and number_per_batch), 
the adversarial learning rate, and the adversarial perturbation epsilons).

While the standard model is the most accurate on unperturbed data ($\\epsilon=0$), 
it is reduced to near-guessing performance by the smallest-magnitude adversarial 
perturbations. By contrast, the robust model's performance decays much more 
gradually in light of perturbations of increasing strength.
        
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
            robustness_comparison_demo(self.params)


def plot_accuracies(params:RobustnessComparisonParams,
                    accuracies_robust:list[float], 
                    accuracies_standard:list[float]) -> None:
    # Plot robustness curves
    fig = plt.figure()
    plt.plot(params.epsilons, accuracies_robust, linewidth=2, marker=".", markersize=10, color="b", label="robust")
    plt.plot(params.epsilons, accuracies_standard, linewidth=2, marker=".", markersize=10, color="r", label="standard")
    plt.xlabel("Epsilon")
    plt.ylabel("Adversarial Accuracy")
    plt.legend()
    plt.xlim([0, 2])
    plt.ylim([0, 1])

    if params.run_streamlit:
        st.pyplot(fig)
    else:
        plt.show()


# Define function for computing adversarial perturbations and adversarial accuracy over multiple batches
def compute_adversarial_accuracy(model, 
                                dataloader, 
                                params: RobustnessComparisonParams, 
                                device: str) -> list[float]:
    accuracies = []

    model = model.to(device)

    def get_stepsize(factor: float, steps: int, epsilon: float) -> float:
        return factor * epsilon / steps

    # Set containers to print evaluation progress
    if params.run_streamlit:
        col1, col2, col3 = st.columns(3)
        status_placeholder1 = col1.empty()
        status_placeholder2 = col2.empty()
        status_placeholder3 = col3.empty()
        my_bar = st.progress(0)

    # Iterate through epsilons
    for EPS in params.epsilons:
        print(f"Epsilon = {EPS}")

        # solver(data: Tensor, target: Tensor)
        solver = partial(
            gradient_ascent,
            model=model,
            optimizer=L2ProjectedOptim,
            lr=get_stepsize(params.learning_rate, params.steps, EPS),
            epsilon=EPS,
            steps=params.steps,
            targeted=False,
            use_best=True,
        )

        data_iter = iter(dataloader)
        accur = 0.0


        for i in tqdm(range(params.num_batches)):
            if params.run_streamlit:
                status_placeholder1.write(f"Evaluating Model: {params.model_name}")
                status_placeholder2.write(f"Epsilon: {EPS}")
                status_placeholder3.write(f"Batch: {i+1}/{params.num_batches}")
                my_bar.progress((i+1)/params.num_batches)

            # Sample next batch
            x, y = next(data_iter)

            # Solve for perturbation
            if EPS != 0:
                x_adv, _ = solver(data=x.to(device), target=y.to(device))
            else:
                x_adv = x

            # Compute accuracy
            x_adv = x_adv.clamp_(0, 1)
            logits_adv = model(x_adv)
                
            y_adv = torch.argmax(logits_adv, axis=1).detach().cpu()
            accur += sum(y == y_adv) / len(x)


        accuracies.append(float(accur / params.num_batches))
    return accuracies


def robustness_comparison_demo(params:RobustnessComparisonParams) -> None:
    
    # CPU or GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load Data
    dataset, dataloader = load_CIFAR10(params.batch_size)
    class_names = dataset.classes

    # Load standard model
    model_standard = load_model(params.model_list[1])
    model_standard.eval()

    # Load robust model
    model_robust = load_model(params.model_list[0])
    model_robust.eval()

    if params.load_results:
        accuracies_standard = [0.9100, 0.1400, 0., 0., 0.]
        accuracies_robust = [0.8600, 0.8200, 0.7200, 0.5100, 0.1300]
    else:
        # Compute adversarial accuracy for standard model
        params.model_name = params.model_list[1]
        accuracies_standard = compute_adversarial_accuracy(
            model_standard,
            dataloader,
            params,
            device=device,
        )

        # Compute adversarial accuracy for robust model
        params.model_name = params.model_list[0]
        accuracies_robust = compute_adversarial_accuracy(
            model_robust,
            dataloader,
            params,
            device=device,
        )

    plot_accuracies(params, accuracies_robust, accuracies_standard)


if __name__=="__main__":
    params = RobustnessComparisonParams()
    params.run_streamlit = False
    params.batch_size = 3
    params.num_batches = 2
    params.load_results = False

    robustness_comparison_demo(params)