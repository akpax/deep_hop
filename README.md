# Deep Hop Rap Lyric Generator Using GPT-J and Multi-Task Learning

## Introduction
This project explores the creation of a Hip Hop Rap Lyric Generator using a fine-tuned GPT-J model. It focuses on generating rap lyrics with an attempt to achieve rhyme and meter. The model was trained on the Pile dataset, known for its abrasive and profane content, reflecting the raw essence of certain rap music styles.

## Objectives
* Data Collection and Preprocessing: Sourcing data from Rap Genius via the Lyrics Genius API, with preprocessing including grapheme-to-phoneme conversion.
* Model Experimentation: The project involved LoRA fine-tuning and experimentation with multi-task learning on GPT-J for lyric generation and phoneme-grapheme conversions.
* Flask Application: Development of a Flask web application for interacting with the AI-generated rap lyrics.

## Model Training and Quantization
* The model was quantized using the bitsandbytes library, reducing its size from 24.2 GB to 5.9 GB, enhancing efficiency.
* Training involved multiple sessions on Google Colab using Tesla V100 GPUs, utilizing the capability to reload from checkpoints for continuous training.
* The LoRA fine-tuning approach was employed to enhance the model's performance without significantly increasing its size.

## Deployment Architecture and Accessibility
The project utilizes Flask and Amazon EC2 with a g4dn.xlarge instance. While the operational costs have led to the website being currently inactive, interested parties can contact me via [LinkedIn](https://www.linkedin.com/in/austin-paxton-98b496165/) for scheduled demonstrations and discussions about generative AI and its applications.

## Cost and Time Consideration
Balancing cost-efficiency and performance, the project's deployment on AWS EC2 was carefully planned. The entire development, including integrating the model into Flask, spanned approximately 3 weeks.

## Evaluation and Future Improvements
* Lyric Quality: The model's ability to rhyme and maintain meter is an ongoing developmental focus.
* User Feedback: Invitations for feedback to refine and enhance the application in future iterations.

## Conclusion
This capstone project represents an ambitious foray into combining machine learning with artistic creativity. It demonstrates the challenges and potential of using natural language processing in the creative domain of Hip Hop lyricism.