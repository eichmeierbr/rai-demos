
import streamlit as st

from functools import partial
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision

from rai_toolbox.perturbations import gradient_ascent, AdditivePerturbation
from rai_toolbox.optim import L2ProjectedOptim
from rai_experiments.models.pretrained import load_model

class AdversarialPerturbParams:
    def __init__(self):
        self.model_list = ('mitll_cifar_l2_1_0.pt', 'mitll_cifar_nat.pt')
        self.model_name = ''
        self.batch_size = 10
        self.eps = 0.5
        self.learning_rate = 2.5
        self.steps = 10

class AdversarialPerturbDemo:
    def __init__(self):
        self.params = AdversarialPerturbParams()

    def get_description(self):
        st.markdown("## Generating Adversarial Perturbations")
        with st.expander("Readme",expanded=True):
            st.markdown(r"""

        This demonstration follows the
        [Generating Adversarial Perturbations Tutorial](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/tutorials/CIFAR10-Adversarial-Perturbations.html)
        from the [rAI-Toolbox](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/)
        
        Solving for adversarial perturbations is a common task in Robust AI. An adversarial perturbation 
        for a given input, $x$, with corresponding label, $y$, can be computed by solving the following 
        optimization objective: """)
                    
            col1, col2, col3 = st.columns(3)
            col2.markdown(r"""
        $$\max_{||\delta||_p<\epsilon} \mathcal{L}\left(f_\theta(x+\delta), y\right) $$ """)

            st.markdown(r"""
        where $\delta$ is the perturbation, $f_\theta$ is a machine learning model parameterized by 
        $\theta$, $\mathcal{L}$ is the loss function (e.g., cross-entropy), $||\delta||_p$ is the 
        $L^p$-norm of the perturbation (for this example we'll be using $p=2$), and $\epsilon$ is the 
        maximum allowable size of the perturbation. 

        This demonstration uses 
        [rai_toolbox.perturbations.gradient_ascent](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/generated/rai_toolbox.perturbations.gradient_ascent.html#rai_toolbox.perturbations.gradient_ascent) 
        to generate $L^2$-norm-constrained adversarial perturbations using projected gradient 
        descent (PGD) for images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 
        dataset, based on either a standard classification model or a robust 
        (i.e., adversarially-trained) model.

        After running the demonstration, three columns of images will appear.
        The first column shows randomly selected images from the CIFAR-10 dataset.
        The number of these images is controllable by the "batch size" input field
        with the image class shown above the image. The second column shows the 
        adversarially perturbed images with the model prediction above the image.
        If the prediction is incorrect, the label will be shown in red. The final
        column shows a 25x exaggeration of the adversarial perturbation.

        Note that when using a non-robust model, the adversarial perturbation has less structure.
        In contrast, the perturbations against the robust model show more strucutre
        around the objects of interest within the image.""")

    def load_parameters(self)-> bool:
        with st.sidebar:
            form = st.form("Adversarial Form")
            
            form.text('Adversarial Perturb Settings')

            self.params.model_name = form.selectbox('Model Selection', self.params.model_list)
            self.params.batch_size = int(form.text_input('Batch Size', '10'))
            self.params.eps = float(form.text_input('Epsilon', '0.5'))
            self.params.learning_rate = float(form.text_input('Learning Rate', '2.5'))
            self.params.steps = int(form.text_input('Steps', '10'))

            return form.form_submit_button("Run Demo")

    def run_demo(self)-> None:
        self.get_description()

        submitted = self.load_parameters()
        
        if submitted:
            adversarial_perturbation_demo(self.params)


def plot_adversarial_images(x, y, class_names, x_adv, y_adv):

    col1, col2, col3 = st.columns(3)
    img_width = 180

    for i in range(len(y)):
        col1.text(class_names[y[i]])
        col1.image(x[i].permute(1, 2, 0).numpy(), width=img_width)

        if y_adv[i].item() != y[i].item():
            original_title = f'<p style="font-family:Courier; color:Red; font-size: 15px;">{class_names[y_adv[i]]}</p>'
            col2.markdown(original_title, unsafe_allow_html=True)
        else:
            original_title = f'<p style="font-family:Courier; color:White; font-size: 15px;">{class_names[y_adv[i]]}</p>'
            col2.markdown(original_title, unsafe_allow_html=True)
        col2.image(x_adv[i].detach().cpu().permute(1, 2, 0).numpy(), width=img_width)

        original_title = f'<p style="font-family:Courier; color:White; font-size: 15px;">perturbation</p>'
        col3.markdown(original_title, unsafe_allow_html=True)
        col3.image(
            (torch.abs(x[i] - x_adv[i].detach().cpu()) * 25)
                .clamp_(0, 1)
                .permute(1, 2, 0)
                .numpy(), 
            width=img_width)

# @st.cache()
def load_CIFAR10(batch_size:int) -> tuple[torchvision.datasets.CIFAR10, torch.utils.data.DataLoader]:
    DATA_DIR = str(Path.home() / ".torch" / "data")

    # Load CIFAR10 test dataset
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )

    # Instantiate a data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return dataset, dataloader


def adversarial_perturbation_demo(params: AdversarialPerturbParams) -> None:
    MODEL_DIR = str(Path.home() / ".torch" / "model")

    # CPU or GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    # Load Data
    dataset, dataloader = load_CIFAR10(params.batch_size)
    class_names = dataset.classes

    x, y = next(iter(dataloader))

    # Load model
    model = load_model(params.model_name)
    model.eval()

    # Define perturbation model
    perturbation_model = AdditivePerturbation(x.shape)

    def get_stepsize(factor: float, steps: int, epsilon: float) -> float:
        return factor * epsilon / steps

    solver = partial(
        gradient_ascent,
        perturbation_model = perturbation_model.to(device),
        optimizer=L2ProjectedOptim,
        lr=get_stepsize(params.learning_rate, params.steps, params.eps),
        epsilon=params.eps,
        steps=params.steps,
        targeted=False,
        use_best=True,
    )

    # Solve for adversarial perturbations to standard model and plot
    x_adv, loss_adv = solver(
        data=x.to(device),
        target=y.to(device),
        model=model.to(device)
    )

    # Compute predictions for adversarial samples
    x_adv = x_adv.clamp_(0, 1)
    logits_adv = model(x_adv)
    y_adv = torch.argmax(logits_adv, axis=1)

    plot_adversarial_images(x, y, class_names, x_adv, y_adv)