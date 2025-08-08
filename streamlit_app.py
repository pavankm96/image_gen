import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from diffusers import StableDiffusionPipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
#from langchain.chains import LLMChain
from PIL import Image
import os
import random

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

# Prompt chains
prompt_chain = 
    prompt=ChatPromptTemplate.from_template("Describe a visually compelling scene based on the idea: {idea}")|llm


recommendation_chain =
    prompt=ChatPromptTemplate.from_template(
        "Based on the image description '{description}', suggest 3 related visual styles or ideas."
    )|llm

# Load Stable Diffusion pipeline on CPU
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to("cpu")

# Load image captioning model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Caption generation function
def generate_multiple_captions(image, num_captions=10):
    captions = []
    for _ in range(num_captions):
        pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values
        caption_model.eval()
        with torch.no_grad():
            output_ids = caption_model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=random.uniform(0.7, 1.0)
            )
        caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

# Streamlit UI
st.title("🎨 AI Image Generator and Captioner")
user_idea = st.text_input("Enter a creative idea for an image:", "a futuristic cityscape at sunset")

if st.button("Generate Image"):
    with st.spinner("Generating prompt..."):
        image_prompt_dict = prompt_chain.invoke(user_idea)
        image_prompt = image_prompt_dict["text"]

    with st.spinner("Generating image..."):
        generated_image = pipe(image_prompt).images[0]
        generated_image.save("generated_image.png")
        st.image(generated_image, caption="Generated Image", use_column_width=True)

    with st.spinner("Generating captions..."):
        captions = generate_multiple_captions(generated_image)
        st.markdown("### 📝 Captions for Generated Image")
        for i, caption in enumerate(captions, 1):
            st.markdown(f"{i}. {caption}")

    with st.spinner("Generating visual style recommendations..."):
        recommendations = recommendation_chain.invoke(captions[0])
        recommended_styles = [style.strip() for style in recommendations.split("\n") if style.strip()]
        st.markdown("### 🎨 Recommended Visual Styles")
        for i, style in enumerate(recommended_styles, 1):
            st.markdown(f"{i}. {style}")

    style_images = []
    for idx, style in enumerate(recommended_styles):
        with st.spinner(f"Generating image for style: {style}"):
            styled_image = pipe(style).images[0]
            filename = f"style_image_{idx+1}.png"
            styled_image.save(filename)
            st.image(styled_image, caption=f"Style {idx+1}: {style}", use_column_width=True)
            style_images.append((style, styled_image, filename))

    st.markdown("### 📝 Caption Options for Style Images")
    for i, (style, image, filename) in enumerate(style_images, 1):
        if st.checkbox(f"Generate 10 captions for style image {i} ('{style}')"):
            with st.spinner(f"Generating captions for '{style}'..."):
                style_captions = generate_multiple_captions(image)
                for j, caption in enumerate(style_captions, 1):
                    st.markdown(f"{j}. {caption}")
