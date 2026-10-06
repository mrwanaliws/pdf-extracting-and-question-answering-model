import pymupdf
from sentence_transformers import SentenceTransformer , util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import streamlit as stm
stm.title("👋 Welcome")
stm.write( """ ### PDF Question Answering Model
Hello! I am an AI model designed to understand the content of your PDF
and answer questions based on the information provided in it.

You can ask me any question related to the PDF, and I will search for
the most relevant information and provide the best answer.
"""
)
pdf_path = "sample.pdf"
doc = pymupdf.open(pdf_path)

text=""
for page in doc:
   page_text = page.get_text()
   lines = page_text.splitlines()
   cleaned_lines = []
   for line in lines:
      if "Educational Test Document | Page" not in line:
         cleaned_lines.append(line)
   text += "\n".join(cleaned_lines) + "\n"
doc.close()

print("text length:" , len(text))
def make_chunks(full_text, chunk_size=440 , overlap=100):
    chunks = []
    start = 0
    text_length = len(full_text)
    step = chunk_size - overlap
    while start < text_length:
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if len(chunk) > 70:
            chunks.append(chunk)
        start += step
    return chunks
chunks = make_chunks(text)
print(f"total chunks : {len(chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks , convert_to_tensor=True)

question = stm.text_input(
   "Ask me  a Question:" ,
   placeholder="example: what is an ISMS"
)
ask_button = stm.button("GET Answer🔍") 
if ask_button:
   if question.strip() == "":
      stm.warning("please enter question")
   else:
      stm.write("searching for the most relevant answer...")
      question_embedding = model.encode(question , convert_to_tensor = True)
      
      similarities = util.cos_sim(question_embedding , chunk_embeddings)[0]
      top_k = 3
      top_indices = torch.topk(
      similarities,
      k=top_k
      ).indices
      print(f"\nQuestion:{question}")
      
      for rank, index in enumerate(top_indices, start=1):
          print(f"\n most relevant chunk {rank} ")
          print(f"simi sscore : {similarities[index].item(): .4f}")
          print(chunks[index])
      
      tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
      qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
      best_answer = ""
      best_answer_score = float("-inf")
      best_chunk_number = 0
      
      results = []
      
      for rank, index in enumerate(top_indices, start=1):
          context = chunks[index]
          retrieval_score =similarities[index].item()
          inputs = tokenizer(
              question,
              context,
              return_tensors="pt",
              truncation = True,
              max_length=512
          )
          sequence_ids = inputs.sequence_ids(0)
          context_positions = [
             i for i, sequence_id in enumerate(sequence_ids)
             if sequence_id == 1
          ]
          print("context token pos:")
          print(context_positions[:10])
          print("num of context tokens:" , len(context_positions))
          with torch.no_grad():
           outputs = qa_model(**inputs)
      
          start_index =outputs.start_logits[0]
          end_index = outputs.end_logits[0]
      
          context_start = min(context_positions)
          context_end = max(context_positions)
      
          context_start_logits = start_index[context_start:context_end + 1]
          context_end_logits = end_index[context_start:context_end + 1]
      
          start_indexes = torch.topk(context_start_logits, k=5).indices
          end_indexes = torch.topk(context_end_logits, k=5).indices
      
          start_indexes = start_indexes + context_start
          end_indexes = end_indexes + context_start
          
          best_start = 0
          best_end = 0
          best_score = float("-inf")
          candidates = []
      
          for start in start_indexes:
             for end in end_indexes:
                 
                 start = int(start)
                 end = int(end)
                 
                 if end < start :
                    continue
                 if end - start + 1 > 30:
                    continue
                  
                 qa_score = (
                    start_index[start].item() +
                     end_index[end].item()
                 )
                 answer_length = end - start + 1
                 length_bonus = 0.15 * min(answer_length, 12)
                 score = qa_score + length_bonus
                 answer_text =  tokenizer.decode(
                   inputs["input_ids"][0][start:end + 1],
                   skip_special_tokens=True
                 ).strip()
                 if answer_text:
                    candidates.append((
                       score, 
                       answer_text,
                       answer_length))
                 if score > best_score:
                   best_score = score
                   best_start = start
                   best_end = end
          candidates.sort(reverse=True, key=lambda x: x[0])
          print("\n top 5 cand")
          for score, answer_text, answer_length in candidates[:5]:
             print(
                f"score: {score:.4f} | "
                f"answer: {answer_text} | "
                f"length: {answer_length}"
                  )
          print("best_start :" , best_start)
          print("best_end :", best_end)
          print("best tokens")
          print(inputs["input_ids"][0][best_start:best_end + 1])
          print("decode")
          print(
             tokenizer.decode(
                inputs["input_ids"][0][best_start:best_end + 1],
                skip_special_tokens=True
             )
          )
      
          answer = tokenizer.decode(inputs["input_ids"][0][best_start:best_end + 1], skip_special_tokens=True).strip()
      
      
          print(f"Question result {rank}")
          print(f"Answer: {answer}")
          print(f"Question answer score: {best_score:.4f}")
          print(f"retrieval score: {retrieval_score:.4f}")
      
          results.append({
             "answer" : answer,
             "qa_score" : best_score,
             "retrieval_score" : retrieval_score,
             "chunk" : rank
            
          }
          )
      qa_scores = [r["qa_score"] for r in results]
      qa_min = min(qa_scores)
      qa_max = max(qa_scores)
      for r in results:
         if qa_max == qa_min:
            r["qa_normalized"] = 0.0
         else:
            r["qa_normalized"] = (
               (r["qa_score"] - qa_min) / (qa_max - qa_min)
            )
         print(
            f"chunk {r['chunk']} | "
            f"Qa: {r['qa_score']:.4f} | "
            f"Qa normalized: {r['qa_normalized']:.4f}"
         )
      retrieval_scores = [r["retrieval_score"] for r in results]
      retrieval_min = min(retrieval_scores)
      retrieval_max = max(retrieval_scores)
      for r in results:
         if retrieval_max == retrieval_min:
            r["retrieval_normalized"] = 0.0
         else:
            r["retrieval_normalized"] = (
               (r["retrieval_score"] - retrieval_min) / (retrieval_max - retrieval_min)
            )
         r["final_score"] = (
            0.6 * r["retrieval_normalized"] + 0.4 * r["qa_normalized"]
         )
         print(
            f"chunk {r['chunk']} | "
            f"retrieval: {r['retrieval_score']:.4f} | "
            f"retrieval_normalized: {r['retrieval_normalized']:.4f} | "
            f"qa_normalized: {r['qa_normalized']:.4f} | "
            f"Final_score: {r['final_score']:.4f}"
         )
      best_result = max(
         results,
         key=lambda result: result["final_score"]
      )
      best_answer = best_result["answer"]
      best_chunk_number = best_result["chunk"]
      best_answer_score = best_result["final_score"]
      print("\n" + "="*50)
      print("final answer :")
      print(f"answer: {best_answer} ")
      print(f"selected chunk : {best_chunk_number}")
      print(f"final score: {best_answer_score:.4f}")
      print("="*50)
      stm.subheader("Answer")
      stm.write(best_answer)

      stm.write(
         f"selected chunk: {best_chunk_number}"
    
      )