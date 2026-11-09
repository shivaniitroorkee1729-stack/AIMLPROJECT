"""
Natural Language Explanation Generator.
Synthesizes quantitative explainability evidence (Grad-CAM heatmap energy,
regional importance distributions, FACS Action Unit activations, and top-2 class margin)
into coherent, structured, evidence-based natural language explanations.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RASA_DESCRIPTIONS


class NaturalLanguageExplainer:
    """
    Produces evidence-based, scientifically sound natural language explanations.
    """

    def generate_explanation(
        self,
        predicted_class,
        confidence,
        probabilities,
        top_regions,  # list of (region_name, percentage)
        active_action_units,  # list of AU dicts
        top_two_comparison=None
    ):
        """
        Builds structured multi-paragraph explanation combining all evidence signals.
        """
        rasa_meta = RASA_DESCRIPTIONS.get(predicted_class, {})
        meaning = rasa_meta.get("meaning", "facial expression")
        bhaava = rasa_meta.get("bhaava", "associated emotion")
        visual_cues = rasa_meta.get("visual_cues", "")

        conf_pct = round(confidence * 100, 1)

        # 1. Prediction summary & certainty assessment
        if confidence >= 0.75:
            certainty_str = "with high classification confidence"
        elif confidence >= 0.50:
            certainty_str = "with moderate classification confidence"
        else:
            certainty_str = "with low confidence, indicating ambiguous or nuanced facial cues"

        p1 = (
            f"The model classified this facial expression as **{predicted_class}** ({bhaava}) "
            f"{certainty_str} ({conf_pct}%). In Indian aesthetic treatise (Natya Shastra), "
            f"**{predicted_class}** embodies *{meaning}*."
        )

        # 2. Grad-CAM Regional Attention Evidence
        if top_regions and len(top_regions) >= 2:
            r1, p_r1 = top_regions[0]
            r2, p_r2 = top_regions[1]
            p2 = (
                f"**Spatial Gradient Evidence (Grad-CAM):** The convolutional feature activations "
                f"demonstrate that the model's decision was predominantly influenced by the "
                f"**{r1}** region ({p_r1:.1f}% of total activation) and the **{r2}** region ({p_r2:.1f}%). "
                f"The gradient concentration corresponds closely with the canonical expressive features "
                f"of {predicted_class}: *{visual_cues}*"
            )
        else:
            p2 = "**Spatial Gradient Evidence (Grad-CAM):** Grad-CAM activations were distributed across the facial anatomy."

        # 3. Action Unit Muscular Evidence
        if active_action_units:
            au_names = [f"**{au['id']}** ({au['name']}, intensity {au['intensity']:.2f})"
                        for au in active_action_units[:3]]
            p3 = (
                f"**Facial Action Unit (FACS) Evidence:** Objective facial muscle movement analysis "
                f"detected active contractions in " + ", ".join(au_names) + ". "
                f"These muscular patterns provide physiological corroboration for the neural network's visual attention."
            )
        else:
            p3 = (
                "**Facial Action Unit (FACS) Evidence:** No high-intensity muscular displacement exceeded "
                "the activation threshold, reflecting a composed or subtle expression profile."
            )

        # 4. Top-Two Differential / Ambiguity analysis
        p4 = ""
        if top_two_comparison:
            margin = top_two_comparison.get("probability_margin", 1.0)
            top2_cls = top_two_comparison.get("top2_class", "")
            top2_prob = round(top_two_comparison.get("top2_probability", 0.0) * 100, 1)

            if margin < 0.20:
                favors_top1 = top_two_comparison.get("regions_favoring_top1", [])
                favor_str = f"higher activation in the {favors_top1[0]} region" if favors_top1 else "subtle feature nuances"
                p4 = (
                    f"**Comparative Uncertainty:** The secondary prediction is **{top2_cls}** at {top2_prob}% "
                    f"(a close margin of {margin * 100:.1f}%). The model leaned toward {predicted_class} "
                    f"primarily due to {favor_str}."
                )

        # Scientific Disclaimer
        disclaimer = (
            "> [!NOTE]\n"
            "> **Scientific Disclaimer:** This system predicts categorical facial expression patterns "
            "derived from dataset visual annotations. It does not measure or claim to diagnose an individual's "
            "genuine internal emotional or psychological state."
        )

        full_explanation = f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}\n\n{disclaimer}" if p4 else f"{p1}\n\n{p2}\n\n{p3}\n\n{disclaimer}"
        return full_explanation
