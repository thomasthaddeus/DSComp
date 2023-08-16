# NLP CommonLit

## Goal of the Competition

The goal of this competition is to assess the quality of summaries written by students in grades 3-12. You'll build a model that evaluates how well a student represents the main idea and details of a source text, as well as the clarity, precision, and fluency of the language used in the summary. You'll have access to a collection of real student summaries to train your model.

Your work will assist teachers in evaluating the quality of student work and also help learning platforms provide immediate feedback to students.

## Context

Summary writing is an important skill for learners of all ages. Summarization enhances reading comprehension, particularly among second language learners and students with learning disabilities. Summary writing also promotes critical thinking, and it’s one of the most effective ways to improve writing abilities. However, students rarely have enough opportunities to practice this skill, as evaluating and providing feedback on summaries can be a time-intensive process for teachers. Innovative technology like large language models (LLMs) could help change this, as teachers could employ these solutions to assess summaries quickly.

There have been advancements in the automated evaluation of student writing, including automated scoring for argumentative or narrative writing. However, these existing techniques don't translate well to summary writing. Evaluating summaries introduces an added layer of complexity, where models must consider both the student writing and a single, longer source text. Although there are a handful of current techniques for summary evaluation, these models have often focused on assessing automatically-generated summaries rather than real student writing, as there has historically been a lack of these types of datasets.

Competition host CommonLit is a nonprofit education technology organization. CommonLit is dedicated to ensuring that all students, especially students in Title I schools, graduate with the reading, writing, communication, and problem-solving skills they need to be successful in college and beyond. The Learning Agency Lab, Vanderbilt University, and Georgia State University join CommonLit in this mission.

As a result of your help to develop summary scoring algorithms, teachers and students alike will gain a valuable tool that promotes this fundamental skill. Students will have more opportunities to practice summarization, while simultaneously improving their reading comprehension, critical thinking, and writing abilities.

## Submission File

For each student_id in the test set, you must predict a value for each of the two analytic measures (described on the Data page). The file should contain a header and have the following format:

student_id,content,wording
000000ffffff,0.0,0.0
111111eeeeee,0.0,0.0
222222cccccc,0.0,0.0
333333dddddd,0.0,0.0

## This is a Code Competition

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

## Dataset Description

The dataset comprises about 24,000 summaries written by students in grades 3-12 of passages on a variety of topics and genres. These summaries have been assigned scores for both content and wording. The goal of the competition is to predict content and wording scores for summaries on unseen topics.

### File and Field Information

`summaries_train.csv` - Summaries in the training set.
`student_id` - The ID of the student writer.
`prompt_id` - The ID of the prompt which links to the prompt file.
`text` - The full text of the student's summary.
`content` - The content score for the summary. The first target.
`wording` - The wording score for the summary. The second target.
`summaries_test.csv` - Summaries in the test set. Contains all fields above except content and wording.
`prompts_train.csv` - The four training set prompts. Each prompt comprises the complete summarization assignment given to students.
`prompt_id` - The ID of the prompt which links to the summaries file.
`prompt_question` - The specific question the students are asked to respond to.
`prompt_title` - A short-hand title for the prompt.
`prompt_text` - The full prompt text.
`prompts_test.csv` - The test set prompts. Contains the same fields as above. The prompts here are only an example. The full test set has a large number of prompts. The train / public test / private test splits do not share any prompts.
`sample_submission.csv` - A submission file in the correct format. See the Evaluation page for details.

## `prompts_train.csv`

| student_id | prompt_question                                                                                         | prompt_title              | prompt_text                                                                                             |
| ---------- | ------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| 39c16e     | Summarize at least 3 elements of an ideal tragedy, as described by Aristotle.                           | On Tragedy                | Chapter 13 As the sequel to what has already been said, we must proceed to consider what the poet ...   |
| 3b9047     | In complete sentences, summarize the structure of the ancient Egyptian system of government. How wer... | Egyptian Social Structure | Egyptian society was structured like a pyramid. At the top were the gods, such as Ra, Osiris, and Is... |
| 814d6b     | Summarize how the Third Wave developed over such a short period of time and why the experiment was e... | The Third Wave            | Background The Third Wave experiment took place at Cubberley High School in Palo Alto, California ...   |
| ebad26     | Summarize the various ways the factory would use or cover up spoiled meat. Cite evidence in your ans... | Excerpt from The Jungle   | With one member trimming beef in a cannery, and another working in a sausage factory, the family had... |
