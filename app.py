import gradio as gr
from model import get_model
from preprocess import preprocess_text

model, tfidf = get_model()

def predict_spam(message):
    cleaned = preprocess_text(message)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    return "Spam" if prediction == 1 else "Not Spam"

interface = gr.Interface(
    fn=predict_spam,
    inputs="text",
    outputs="text",
    title="Spam Detection App",
    description="Enter a message to check if it is Spam or Not Spam"
)

if __name__ == "__main__":
    interface.launch()