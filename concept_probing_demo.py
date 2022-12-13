import streamlit as st

import torch
import torchvision

from functools import partial

from rai_toolbox.perturbations import gradient_ascent
from rai_experiments.models.pretrained import load_model
from rai_toolbox.optim import L1qFrankWolfe 
from rai_experiments.utils.imagenet_labels import IMAGENET_LABELS



class ContentProbingParams:
    def __init__(self):
        self.model_list = ('mitll_imagenet_l2_3_0.pt', 'imagenet_nat.pt')
        self.model_name = ''
        self.eps = 7760
        self.learning_rate = 1.0
        self.steps = 45
        self.targets = []
        self.q = 0.975
        self.dq = 0.05


class ContentProbingDemo:
    def __init__(self):
        self.params = ContentProbingParams()

    def get_description(self) -> None:
        st.markdown("## Concept Probing Using Sparse Perturbations")

        with st.expander("Readme",expanded=True):
            st.markdown("""

This demonstration follows the 
[Concept Probing Using Sparse Perturbations Tutorial](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/tutorials/ImageNet-Concept-Probing.html)
from the [rAI-Toolbox](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/)

This demonstration shows how the 
[gradient_ascent](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/generated/rai_toolbox.perturbations.gradient_ascent.html#rai_toolbox.perturbations.gradient_ascent) 
perturbation can be repurposed with a different optimizer 
([L1qFrankWolfe](https://mit-ll-responsible-ai.github.io/responsible-ai-toolbox/generated/rai_toolbox.optim.L1qFrankWolfe.html)) 
to execute a task called concept probing.
The method of concept probing was first reported in the paper [Controllably Sparse Perturbations of Robust Classifiers 
for Explaining Predictions and Probing Learned 
Concepts](https://diglib.eg.org/bitstream/handle/10.2312/mlvis20211072/001-005.pdf?sequence=1&isAllowed=y).

In Concept Probing, we'll be starting with a random noise image and optimizing its perturbation towards a class of 
interest, in an attempt to visualize what the model has learned for that class. Following the approach proposed by the 
authors of [this paper](https://diglib.eg.org/bitstream/handle/10.2312/mlvis20211072/001-005.pdf?sequence=1&isAllowed=y), 
we utilize the $L^{1-q}$ Frank-Wolfe optimizer to solve for sparse perturbations that are more interpretable by humans.

To run this demonstration, select the model you would like to use for concept probing. mitll_imagenet_l2_3_0.pt is a 
robust model trained with the rAI-Toolbox and imagenet_nat.pt is a traditionally trained model. Then, select which 
target classes you want to visualize Three target classes are provided, but you can remove them, or add as many other 
classes as desired. After pressing "Run Demo", you will see two columns of images generated for each target class. The 
first image is a random noise image that is used for each target example. The next column is the target class the model 
"sees" within the random noise

Note that the concepts from the robust model are much more pronounced and look like the target class to the human eye.""")


    def load_parameters(self) -> bool:
        with st.sidebar:
            form = st.form("Perturb Form")
            
            st.text('Content Probing Settings')

            self.params.model_name = form.selectbox('Model Selection', self.params.model_list)
            self.params.learning_rate = float(form.text_input('Learning Rate', '1.0'))
            self.params.eps = int(form.text_input('Epsilon', '7760'))
            self.params.q = float(form.text_input('q', '0.975'))
            self.params.dq = float(form.text_input('dq', '0.05'))
            self.params.steps = int(form.text_input('Steps', '45'))

            def target_class_formatter(key:int) -> str:
                return f"{key}-{IMAGENET_LABELS[key]}"
            self.params.targets = form.multiselect("Target Classes", IMAGENET_LABELS, 
                                                        [75,17,965],
                                                        target_class_formatter)


            return form.form_submit_button("Run Demo")


    def run_demo(self)-> None:
        self.get_description()

        submitted = self.load_parameters()

        if submitted:
            concept_probing_demo(self.params)



def concept_probing_demo(params:ContentProbingParams) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  

    # Load the desired model
    model = load_model(params.model_name)
    model.eval();

    # Setup the solver
    solver_L1FW = partial(
        gradient_ascent,
        optimizer=L1qFrankWolfe,
        # optimizer options
        lr=params.learning_rate,
        epsilon=params.eps,
        q=params.q,
        dq=params.dq,
        # solver options
        steps=params.steps,
        targeted=True,
        use_best=False
    )

    # Random noise image
    init_noise = torch.randn([1, 3, 224, 224])
    init_noise = init_noise - init_noise.min()
    init_noise = init_noise / init_noise.max()

    col1, col2 = st.columns(2)

    for i in range(len(params.targets)):
        # optimize loss towards target class
        target = torch.tensor([params.targets[i]])
        
        # run solver
        x_concept, _ = solver_L1FW(
            model=model.to(device),
            data=init_noise.to(device),
            target=target.to(device),
        )

        # plot
        col1.text("Noise")
        col1.image(init_noise[0].permute(1,2,0).numpy())

        col2.text(IMAGENET_LABELS[params.targets[i]].partition(",")[0])
        col2.image(x_concept[0].detach().cpu().clamp_(0,1).permute(1,2,0).numpy())