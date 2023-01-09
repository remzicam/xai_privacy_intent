"""XAI for Transformers Intent Classifier App."""

from collections import Counter
from itertools import count
from operator import itemgetter
from re import DOTALL, sub

import streamlit as st
from plotly.express import bar
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from transformers_interpret import SequenceClassificationExplainer

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
hide_plotly_bar = {"displayModeBar": False}
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
repo_id = "remzicam/privacy_intent"
task = "text-classification"
title = "XAI for Intent Classification and Model Interpretation"
st.markdown(
    f"<h1 style='text-align: center; color: #0068C9;'>{title}</h1>", unsafe_allow_html=True
)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    """
    It loads the model and tokenizer from the HuggingFace model hub, and then creates a pipeline object
    that can be used to make predictions. Also, it creates model interpretation object.
    
    Returns:
      the privacy_intent_pipe and cls_explainer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    privacy_intent_pipe = pipeline(
        task, model=model, tokenizer=tokenizer, return_all_scores=True
    )
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    return privacy_intent_pipe, cls_explainer


privacy_intent_pipe, cls_explainer = load_models()


def label_probs_figure_creater(input_text:str):
    """
    It takes in a string, runs it through the pipeline, and returns a figure and the label with the
    highest probability
    
    Args:
      input_text (str): The text you want to analyze
    
    Returns:
      A tuple of a figure and a string.
    """
    outputs = privacy_intent_pipe(input_text)[0]
    sorted_outputs = sorted(outputs, key=lambda k: k["score"])
    prediction_label = sorted_outputs[-1]["label"]
    fig = bar(
        sorted_outputs,
        x="score",
        y="label",
        color="score",
        color_continuous_scale="rainbow",
        width=600,
        height=400,
    )
    fig.update_layout(
        title="Model Prediction Probabilities for Each Label",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(  # attribures for x axis
            showline=True,
            showgrid=True,
            linecolor="black",
            tickfont=dict(family="Calibri"),
        ),
        yaxis=dict(  # attribures for y axis
            showline=True,
            showgrid=True,
            linecolor="black",
            tickfont=dict(
                family="Times New Roman",
            ),
        ),
        plot_bgcolor="white",
        title_x=0.5,
    )
    return fig, prediction_label


def xai_attributions_html(input_text: str):
    """
    1. The function takes in a string of text as input.
    2. It then uses the explainer to generate attributions for each word in the input text.
    3. It then uses the explainer to generate an HTML visualization of the attributions.
    4. It then cleans up the HTML visualization by removing some unnecessary HTML tags.
    5. It then returns the attributions and the HTML visualization
    
    Args:
      input_text (str): The text you want to explain.
    
    Returns:
      the word attributions and the html.
    """

    word_attributions = cls_explainer(input_text)
    #remove special tokens
    word_attributions = word_attributions[1:-1]
    # remove strings shorter than 1 chrachter
    word_attributions = [i for i in word_attributions if len(i[0]) > 1]
    html = cls_explainer.visualize().data
    html = html.replace("#s", "")
    html = html.replace("#/s", "")
    html = sub("<th.*?/th>", "", html, 4, DOTALL)
    html = sub("<td.*?/td>", "", html, 4, DOTALL)
    return word_attributions, html+"<br>"


def explanation_intro(prediction_label: str):
    """
    generates model explanaiton html markdown from prediction label of the model.

    Args:
      prediction_label (str): The label that the model predicted.
    
    Returns:
      A string
    """
    return f"""<div style="background-color: lightblue;
  color: rgb(0, 66, 128);">The model predicted the given sentence as <span style="color: black"><strong>'{prediction_label}'</strong></span>.
    The figure below shows the contribution of each token to this decision.
    <span style="color: darkgreen"><strong> Green </strong></span> tokens indicate a <strong>positive </strong> contribution, while <span style="color: red"><strong> red </strong></span> tokens indicate a <strong>negative</strong> contribution.
    The <strong>bolder</strong> the color, the greater the value.</div><br>"""


def explanation_viz(prediction_label: str, word_attributions):
    """
    It takes in a prediction label and a list of word attributions, and returns a markdown string that contains
    the word that had the highest attribution and the prediction label
    
    Args:
      prediction_label (str): The label that the model predicted.
      word_attributions: a list of tuples of the form (word, attribution score)
    
    Returns:
      A string
    """
    top_attention_word = max(word_attributions, key=itemgetter(1))[0]
    return f"""The token **_'{top_attention_word}'_** is the biggest driver for the decision of the model as **'{prediction_label}'**"""


def word_attributions_dict_creater(word_attributions):
    """
    It takes a list of tuples, reverses it, splits it into two lists, colors the scores, numerates
    duplicated strings, and returns a dictionary
    
    Args:
      word_attributions: This is the output of the model explainer.
    
    Returns:
      A dictionary with the keys "word", "score", and "colors".
    """
    word_attributions.reverse()
    words, scores = zip(*word_attributions)
    # colorize positive and negative scores
    colors = ["red" if x < 0 else "lightgreen" for x in scores]
    # darker tone for max score
    max_index = scores.index(max(scores))
    colors[max_index] = "darkgreen"
    # numerate duplicated strings
    c = Counter(words)
    iters = {k: count(1) for k, v in c.items() if v > 1}
    words_ = [x + "_" + str(next(iters[x])) if x in iters else x for x in words]
    # plotly accepts dictionaries

    return {
        "word": words_,
        "score": scores,
        "colors": colors,
    }


def attention_score_figure_creater(word_attributions_dict):
    """
    It takes a dictionary of words and their attention scores and returns a bar graph of the words and
    their attention scores with specified colors.
    
    Args:
      word_attributions_dict: a dictionary with keys "word", "score", and "colors"
    
    Returns:
      A figure object
    """
    fig = bar(word_attributions_dict, x="score", y="word", width=400, height=500)
    fig.update_traces(marker_color=word_attributions_dict["colors"])
    fig.update_layout(
        title="Word-Attention Score",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(  # attribures for x axis
            showline=True,
            showgrid=True,
            linecolor="black",
            tickfont=dict(family="Calibri"),
        ),
        yaxis=dict(  # attribures for y axis
            showline=True,
            showgrid=True,
            linecolor="black",
            tickfont=dict(
                family="Times New Roman",
            ),
        ),
        plot_bgcolor="white",
        title_x=0.5,
    )

    return fig


form = st.form(key="intent-form")
input_text = form.text_area(
    label="Text",
    value="At any time during your use of the Services, you may decide to share some information or content publicly or privately.",
)
submit = form.form_submit_button("Submit")

if submit:
    label_probs_figure, prediction_label = label_probs_figure_creater(input_text)
    st.plotly_chart(label_probs_figure, config=hide_plotly_bar)
    explanation_general = explanation_intro(prediction_label)
    st.markdown(explanation_general, unsafe_allow_html=True)
    with st.spinner():
      word_attributions, html = xai_attributions_html(input_text)
      st.markdown(html, unsafe_allow_html=True)
      explanation_specific = explanation_viz(prediction_label, word_attributions)
      st.info(explanation_specific)
      word_attributions_dict = word_attributions_dict_creater(word_attributions)
      attention_score_figure = attention_score_figure_creater(word_attributions_dict)
      st.plotly_chart(attention_score_figure, config=hide_plotly_bar)
