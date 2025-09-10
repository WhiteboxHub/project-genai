
# import torch
# from PIL import Image
# from transformers import AutoTokenizer, AutoModelForCausalLM

# def encode_image(image_path, prompt="Describe this image."):
#     # Load model and tokenizer
#     model_id = "apple/FastVLM-7B"
#     tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     # Prepare chat template
#     messages = [{"role": "user", "content": "<image>\n" + prompt}]
#     rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     pre, post = rendered.split("<image>", 1)
#     pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
#     post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
#     IMAGE_TOKEN_INDEX = -200
#     img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
#     input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
#     attention_mask = torch.ones_like(input_ids, device=model.device)
#     # Preprocess image
#     img = Image.open(image_path).convert("RGB")
#     px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
#     px = px.to(model.device, dtype=model.dtype)
#     # Generate output
#     with torch.no_grad():
#         out = model.generate(
#             inputs=input_ids,
#             attention_mask=attention_mask,
#             images=px,
#             max_new_tokens=128,
#         )
#     return tok.decode(out, skip_special_tokens=True)