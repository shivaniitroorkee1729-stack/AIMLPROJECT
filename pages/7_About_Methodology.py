"""
Page 7: About / Methodology — Scientific Background, Formulations, Stack, and Limitations
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.visualization import get_custom_css, THEME

st.set_page_config(page_title="About & Methodology — NAVARASA AI", page_icon="📜", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">RESEARCH SPECIFICATION</span>
        <h1 style="margin-top: 10px;">About & Research Methodology</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Theoretical foundations, algorithmic formulations, architectural designs,
            scientific limitations, and technological infrastructure of the Navarasa AI project.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_theory, tab_methods, tab_xai, tab_stack, tab_limits = st.tabs([
    "🎭 Classical Aesthetic Theory",
    "🧠 Model Architectures",
    "🔍 Explainable AI Formulations",
    "💻 Technology Stack",
    "⚠️ Scientific Limitations"
])

with tab_theory:
    st.markdown("## Classical Aesthetic Theory: The Navarasas")
    st.markdown(
        """
        The **Rasa Siddhanta** formulated by sage Bharata Muni in the seminal treatise *Natya Shastra* 
        (circa 200 BCE – 200 CE) posits that human dramatic art evokes nine primary aesthetic experiences (*Rasas*). 
        Each Rasa corresponds to a permanent psychic state (*Sthayi Bhaava*), communicated through transient emotional 
        determinants (*Vibhaavas*), physical expressions and gestural gestures (*Anubhaavas*), and involuntary somatic 
        reactions (*Sattvika Bhaavas*).

        ### The Nine Canonical Rasas:
        1. **Shringara (Rati):** Love, romance, beauty, and devotion. Soft smile, gentle gaze.
        2. **Hasya (Hasa):** Mirth, joy, laughter, and comedic amusement. Elevated cheek apples (AU6), broad smile (AU12).
        3. **Karuna (Shoka):** Pathos, compassion, sorrow, and sympathetic grief. Inner brow elevation (AU1), downturned mouth corners (AU15).
        4. **Raudra (Krodha):** Fury, wrath, and indignation. Strongly furrowed brow (AU4), intense glaring eyes, clenched lips.
        5. **Veera (Utsaha):** Valor, heroism, chivalry, and resolute determination. Poised facial tension, high elevated brow, firm jaw.
        6. **Bhayanaka (Bhaya):** Dread, fright, and horror. Widened eye aperture exposing sclera (AU5), raised eyebrows (AU1+2).
        7. **Bibhatsa (Jugupsa):** Aversion, disgust, and revulsion. Wrinkled nasal bridge (AU9), raised upper lip (AU10).
        8. **Adbhuta (Vismaya):** Wonder, astonishment, and awe. Widened circular eyes, arched brows, open dropped jaw (AU26).
        9. **Shanta (Sama):** Serenity, tranquility, and meditative peace. Composed neutral lips, absence of muscular strain.
        """
    )

with tab_methods:
    st.markdown("## Dual Machine Learning Methodologies")
    st.markdown(
        """
        ### Pipeline A: CLIP ViT-B/32 + Logistic Regression
        - **Vision Transformer Backbone:** Uses OpenAI's Vision Transformer (ViT-B/32) encoder frozen during training.
        - **Embedding Extraction:** Passes 224x224 RGB face images to extract 512-dimensional, L2-normalized feature vectors.
        - **Classifier:** L2-regularized multinomial Logistic Regression trained with L-BFGS solver.
        - **Hyperparameter Optimization:** Regularization strength parameter $C$ tuned across $[0.01, 10.0]$ on the validation split.

        ### Pipeline B: EfficientNet-B0 (Two-Stage Fine-Tuning)
        - **Backbone:** EfficientNet-B0 with depth compound scaling, initialized from ImageNet-1K pretrained weights.
        - **Head Replacement:** Global Average Pooling ➔ Dropout ($p=0.4$) ➔ 9-class Linear output layer.
        - **Phase 1 Training:** Frozen backbone; only the 9-class classification head is optimized using AdamW ($\eta = 10^{-3}$) for initial alignment.
        - **Phase 2 Fine-Tuning:** Unfreezing the top two MBConv stages; fine-tuning with a lower learning rate ($\eta = 10^{-4}$) and Cosine Annealing schedule.
        - **Loss Function:** Cross-Entropy loss with label smoothing ($\epsilon = 0.05$) to mitigate overconfidence.
        """
    )

with tab_xai:
    st.markdown("## Explainable AI (XAI) Mathematical Formulations")
    st.markdown(
        r"""
        ### 1. Gradient-weighted Class Activation Mapping (Grad-CAM)
        To compute class-specific localization heatmaps for Navarasa class $c$:
        
        $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
        
        where:
        - $y^c$ is the unnormalized classification score (logit) for class $c$.
        - $A^k$ is the $k$-th feature map of the final convolutional layer (shape $u \times v$).
        - $Z = u \cdot v$ is the spatial area of the feature map.
        
        The coarse heat map is obtained via a weighted linear combination followed by ReLU:
        
        $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

        ### 2. Facial Region Proportional Energy Analysis
        Given binary spatial mask $M_r$ for anatomical region $r \in \{\text{Eyes, Eyebrows, Nose, Mouth, Cheeks, Forehead}\}$:
        
        $$E_r = \sum_{(x,y) \in M_r} L_{\text{Grad-CAM}}^c(x, y), \quad P_r = \frac{E_r}{\sum_{r'} E_{r'}} \times 100\%$$

        ### 3. FACS Action Unit Muscular Measurements
        Extracts 52 facial blendshape coefficients from dense 478-point 3D landmark mesh. Coefficients are mapped 
        to standardized Facial Action Units (AU1: Inner Brow Raiser, AU4: Brow Lowerer, AU6: Cheek Raiser, 
        AU9: Nose Wrinkler, AU12: Smile Puller, AU15: Lip Corner Depressor, etc.).
        """
    )

with tab_stack:
    st.markdown("## Technological Infrastructure & Libraries")
    st.markdown(
        """
        - **Deep Learning Framework:** PyTorch 2.x, Torchvision
        - **Foundation Models:** OpenCLIP (`ViT-B-32`, OpenAI pretrained)
        - **Computer Vision & Mesh:** MediaPipe Tasks (`face_landmarker.task`), OpenCV 4.x
        - **Machine Learning & Evaluation:** Scikit-learn 1.x, NumPy, SciPy
        - **Data Processing:** Pandas, Pillow
        - **Visualization:** Matplotlib, Seaborn, Altair
        - **Web Application:** Streamlit multi-page framework
        - **Deployment Target:** Streamlit Community Cloud / Docker / Linux VPS
        """
    )

with tab_limits:
    st.markdown("## Scientific Limitations & Ethical Scope")
    st.markdown(
        """
        > [!IMPORTANT]
        > **Ethical & Epistemological Boundaries:**
        > 1. **Visual Expression vs Internal Affect:** The model identifies dataset-defined facial appearance categories 
        >    associated with Indian classical drama expressions. It does **not** possess telepathic access to a person's 
        >    genuine internal feelings or psychological state.
        > 2. **Cultural Stylization in Navarasa:** Classical performers often exaggerate facial features (e.g. eye widening 
        >    in *Kathakali* or *Bharatanatyam*) beyond everyday naturalistic expressions.
        > 3. **Grad-CAM Correlative Nature:** Regional activation percentages measure neural attention density, 
        >    not causal necessity. Occlusion sensitivity tests provide complementary perturbation evidence.
        > 4. **Action Unit Fallbacks:** If adverse lighting or extreme head pose obscures facial landmarks, 
        >    Action Unit extraction falls back gracefully to proportional geometric bounds.
        """
    )
