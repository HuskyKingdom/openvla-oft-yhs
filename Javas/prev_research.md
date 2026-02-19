
# ICML Research on VL-only VLA

###### tags: `Research`

<!-- 98.9 & 98.5 & 98.2 & 97.1 -->
- Auto-generated Table of Content
[ToC]

___


## Previous Observations

We found that our baseline model is highly overfitted, so that the instructions were ignored..

Given raw instruction:

`"put both the alphabet soup and the cream cheese box in the basket_object"`


<div style="display: flex; justify-content: center; align-items: center;">
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/H1b5fNKhle.gif" alt="Before" width="300px" />
<figcaption>Raw | <span style="color: green;">Success</span></figcaption>
  </figure>
<figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/SJWs7NKhle.gif" alt="After" width="300px" />
    <figcaption>Raw: Object Shifted | <span style="color: red;">Failed</span></figcaption>
  </figure>
</div>!


<div style="display: flex; justify-content: center; align-items: center;">
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/S1z6zEt3gx.gif" alt="After" width="300px" />
    <figcaption>soup -> tomato source | <span style="color: green;">Success</span></figcaption></figcaption>
  </figure>
    <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/S1z6zEt3gx.gif" alt="After" width="300px" />
    <figcaption>Empty Prompt | <span style="color: green;">Success</span></figcaption></figcaption>
  </figure>
</div>!


We wonder if this is only the case of OpenVLA, therefore we evluated the same problem on `Pi0.5` (official LIBERO docker released several week ago..), it shows the same behavior:

![4eacba390be72646d178c9435ad34e77](https://hackmd.io/_uploads/SkIUYVK3ge.png)


We then virsulize the attention socres of the model, we wonder this due to the few attentions assigned with instructions:

*showing attention scores after `softmax()` and were averaged across heads.



<div style=" justify-content: center; align-items: center;">
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/ryK1BrY3xl.png" alt="After" width="300px" />
    <figcaption>Raw: T=0</figcaption></figcaption>
    </figure>
    <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/rJIdSHK2ee.png" alt="After" width="300px" />
    <figcaption>Raw: T=56</figcaption>
  </figure>
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/HyvDIHKhlg.png" alt="After" width="300px" />
    <figcaption>Empty Prompt: T=0</figcaption>
  </figure>
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/rk1rIHFhle.png" alt="After" width="300px" />
    <figcaption>Empty Prompt: T=56</figcaption>
  </figure>
</div>!


We do see some attentions were put into the prompt, but note that given a instruction, the VLM propmt tools will be called to wrap this to the following format in order to match the pre-trained form:

`"In: What action should the robot take to {instruction}?  Out:"`

Where this prompt is **identical across episodes**, and the warping prompt is **identical across all samples**. Given that statistically the prompt tokens only contributes a small parts to the model input compare to patches, we assume that such facts causes the model to ignore/noiselize the instructions, this might be the reason why current VLA models are only **vision-based overfitting**.


## SR Eval

Original Trained Model:
| Method                      | Long SR (%) | Spatial SR (%) |
| --------------------------- | -----------:| --------------:|
| Original OpenvlaOFT         |        94.4 |           97.4 |
| w/o Instruction             |        85.8 |           82.6 |
| w/o Wraping                 |         0.0 |            0.0 |
| w/o Instruction and Wraping |         0.0 |            0.0 |

Rephrase VLA Model (fix wraping text in eval):
| Method                      | Long SR (%) | Spatial SR (%) |
| --------------------------- | -----------:| --------------:|
| Original OpenvlaOFT         |        93.2 |          96.8 |
| w/o Instruction   EF          |        82.4 |           85.4 |
| w/o Wraping       NT          |        0.0 |            0.0 |
| w/o Instruction and Wraping ET |         0.0 |            0.0 |


Prompt example:

- Original OpenvlaOFT : 
`In: What action should the robot take to put both the alphabet soup and the cream cheese box in the basket_object?  Out:`
- w/o Instruction  : 
`In: What action should the robot take to empty?  Out:`
- w/o Wraping : 
`put both the alphabet soup and the cream cheese box in the basket_object`
- w/o Instruction and Wraping : 
`empty`

[Bias on modality]



## LIBERO-Plus

[LIBERO-Plus: In-depth Robustness Analysis of Vision-LanguageAction Models](https://arxiv.org/pdf/2510.13626) studies the same question, in their work, they have identified several aspect that effects the performance of the existing model:

![image](https://hackmd.io/_uploads/BJk7pCceZe.png)


And hence propose an augumented data namely `LIBERO-Plus` so that the model trained has better generalizability:

![image](https://hackmd.io/_uploads/Hk-KCCqe-l.png)

### So, what is happning?

However, in their paper they do point out that the current VLA models are ignoring the instruction data completly, but did not propose any solution. In the paper, they hypotheses the following to explain the observation: 

- (1) The model may possess strong generalization capabilities in the language domain, allowing it to remain robust even when instructions are perturbed. (X) <p style="color: red;">Not Possible.</p>

    <p style="color: red;"> Both their experiments and ours shown the models can success even with empty instruction: </p>
    
    ![image](https://hackmd.io/_uploads/S1z6zEt3gx.gif)


- (2) The model may extract limited keywords from the input instruction for matching and decision-making, rather than genuinely understanding the full semantic structure. 

    <p style="color: red;">Unlikely, because their perturbations include a commonsense subclass that performs keyword commonsense rewrite, yet the performance drop remains nearly negligible.</p> 

- (3) The model may not fully utilize the language modality, instead relying primarily on visual or other nonlinguistic signals to complete tasks. In such a scenario, language inputs would be functionally redundant, and even significant perturbations would have minimal impact. 

    <p style="color: green;"> Most likely. </p>


### What are the causes?


I have suspected the following as the potention cases of this problem:

![image](https://hackmd.io/_uploads/ryK1BrY3xl.png) 
- Multimodal bias: instruction token too less? 

    <p style="color: red;"> This is unlikely. </p>
    <p style="color: red;"> 1. The definition of attention underlines that the ability of the model to highlight the important features regardless of its sequence length, attention process them in paralell. Additionally, </p>
    
    <p style="color: red;">2.Video-based VLM, as a popular field, is a similar VLM application, they process the same amount of instruction tokens perfectly.</p>
    
    

I suspect the reason is that: 

**Directly apply VLM in VLA is naive.** 

The instruction describs the **abstract desire** of the **whole trajectory** in **3D space.** Directly apply VLM on embodied action agent would allow it to only overfitts the expert trajectory.

--- 

**Abstract Desire**: the model might not understanding the instruction since it is not a concrete plan.

**Whole Trajectory**: agent failed to inference the outcome of motion, input given is not a direct relevent mapping of output, environment is somehow partially observable, but in Video-VLM it is fully observable.

**3D space**: model has no knowledge on 3D spatial environment.



---
I propose to:

Expert episode -> model -(text decoder)> plan [PRM]

PRM can refer to paper [Let’s Verify Step by Step - OpenAI](https://arxiv.org/pdf/2305.20050)

Expert episode -> model -(action decoder)> action



## Action Plan Dataset (APD)


Model used: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

![image](https://hackmd.io/_uploads/HJeD97nxbg.png)


Given prompt:

```
prompt = """                                                                                 You are an expert robotic task planner. Given a single image and a natural-language instruction, produce a detailed step-by-step action plan for the robot.                               
STRICT RULES YOU MUST FOLLOW:                                                                1. Output must be a JSON list only.
2. The final step must always return the robot to its initial position (e.g., reset base/arm to neutral home pose).
3. Before ANY action you output, must be an simple low-level action that can be done in one motion.
4. Each item must have:
   - "step": integer starting from 1
   - "subgoal": concise actionable description
   - "expected_effect": observable change in the environment
5. DO NOT output any explanation. Output the JSON list only.
6. Always describe spatial relation (e.g. top, left, right,...)

Here is an example format:

[
  {\\"step\\": 1, \\"subgoal\\": \\"Move gripper on top of the cabinet\\", \\"expected_effect\\": \\"robot base positioned on top of the cabinet\\"},
  {\\"step\\": 2, \\"subgoal\\": \\"Reach the handle with the right arm\\", \\"expected_effect\\": \\"gripper aligned with the cabinet handle\\"},
  {\\"step\\": 3, \\"subgoal\\": \\"Pull the cabinet door open\\", \\"expected_effect\\": \\"cabinet door fully opened\\"},
  {\\"step\\": 4, \\"subgoal\\": \\"Move hand into the drawer\\", \\"expected_effect\\": \\"gripper inside the drawer\\"},
  {\\"step\\": 5, \\"subgoal\\": \\"Pick up the black bowl\\", \\"expected_effect\\": \\"robot holds the black bowl securely\\"},
  {\\"step\\": 6, \\"subgoal\\": \\"Place bowl on the plate\\", \\"expected_effect\\": \\"bowl stably placed on the plate surface\\"},
  {"step": 7, "subgoal": "Return to initial position", "expected_effect": "robot back at neutral home pose"}
]

Now, based on the provided image and natural-language instruction, generate the action plan.
Output only the JSON list following the specified format.

The instruction is \\"put the white mug on the plate and put the chocolate pudding to the right of the plate \\"

Output util the instruction is all done.
"""
```

Generated plan:

```
[
  {"step": 1, "subgoal": "Move gripper to the white mug", "expected_effect": "gripper positioned above the white mug"},
  {"step": 2, "subgoal": "Pick up the white mug", "expected_effect": "robot holds the white mug securely"},
  {"step": 3, "subgoal": "Move gripper to the plate", "expected_effect": "gripper positioned above the plate"},
  {"step": 4, "subgoal": "Place the white mug on the plate", "expected_effect": "white mug stably placed on the plate surface"},
  {"step": 5, "subgoal": "Move gripper to the chocolate pudding", "expected_effect": "gripper positioned above the chocolate pudding"},
  {"step": 6, "subgoal": "Pick up the chocolate pudding", "expected_effect": "robot holds the chocolate pudding securely"},
  {"step": 7, "subgoal": "Move gripper to the right of the plate", "expected_effect": "gripper positioned to the right of the plate"},
  {"step": 8, "subgoal": "Place the chocolate pudding to the right of the plate", "expected_effect": "chocolate pudding stably placed to the right of the plate surface"},
  {"step": 9, "subgoal": "Return to initial position", "expected_effect": "robot back at neutral home pose"}
]

```


I have now generated the plans for all instruction in LIBERO dataset.

[APD_plans.json](https://drive.google.com/file/d/1xY7H_55rkyXo2Ht08c5bMarqwCCVY1nV/view?usp=sharing)


## What we do on APD

We need to design a method that perform the following elegantly, leveraging VLM capbilities better...

1. How can we supervise the planning?

2. How to excute the plan?

- (low-level control policy) [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/pdf/2204.01691).
- (world model) [3DFlowAction: Learning Cross-EmbodimentManipulation from 3D Flow World Model](https://arxiv.org/pdf/2506.06199).

3. How to recap?

- VL Model (clip..) to verify current observation and `expected_effect`.


<p style="color: green;"> APD serves as the supervise signal to enable instruction-oriented computing (thinking). </p>


### ECoT (CoRL 2024)

Based on OpenVLA, which they extend the output to include planing and reasoning steps, supervised by auto-generated dataset.

![image](https://hackmd.io/_uploads/rJgZ-yQWbl.png)

<p style="color: green;"> First work that explores CoT in Embodiment! With a good large scale action CoT dataset. </p>

<p style="color: red;"> Limitations: <br> (1) VLA auto-agressive prediction is slow with this many tokens. <br> (2) The tasks are generally not thinking-intensive. </p>


### VLM2VLA (ICLR 2025)
![image](https://hackmd.io/_uploads/BkIYX1QZZe.png)

<p style="color: green;"> Agent behaviour extracted from VLM directly trained with LoRA finetune.</p>

<p style="color: red;"> Limitations: <br> Not an action model.</p>


### Hybrid Training (HyT) (ICLR 2025)

![image](https://hackmd.io/_uploads/SJnICkmZbe.png)


<p style="color: green;"> Enable CoT with efficiency. Using a modality token to ebable hybird training, during inference use only <action> token. Proposal: improve of ECoT is not the thinking progress during inference but model seeing more in training. </p>
    
    
## Adaptive MoE VLA
    

> VLM4VLA: VLM’s general capabilities are poor predictors of its downstream task performance... 
    
I argue that the thinking is either not ignorable at all during inference, or presistently needed, it should rather be an important step to take for several key frames, which the policy is changing. This intuitive also fits the human logic while dealing with such task.

- Output thinkings is benifitial because it increase the explanability and make human-in-the-loop approchese possible, should be helpful.
- However always output it would cause a large latency during inference, we want to showcase that **thinking is important but natually not always needed during each timestep.**
    
    
![image](https://hackmd.io/_uploads/SJJ_c8BZZl.png)


It is my intuitive to propose an Adaptive VLA via MoE based approach, where it dynamically determin whether in current step, the thinking is needed. 
    
![image](https://hackmd.io/_uploads/rkZ9GIS-bx.png)

    
### Action Recap & Human-in-the-loop Correction  

In APD we have the `expected effect` that might be utilized to enable the recap ability of the model:
    
![image](https://hackmd.io/_uploads/rkdgHOB--e.png)

    

The following figure shows an evaluation on how well the VL model (e.g. sigclip2) might determe the success of some action.
    
![image](https://hackmd.io/_uploads/SJJG4uHZWg.png)

This can be serves as an additional supervise signal for our agent.
    
    
    
## Current Method Overview
    
    
In terms of how instruction is processed in VLA system, we can summerize the current works as the follow:
    
![图片](https://hackmd.io/_uploads/HkPfyEt--l.png)

We propose to aligning the vision-language semantic understanding with lower level specific spatial scale:
    
- fine-grained the action learning into furtherly lower level, trying to eable the model learns more directly mapping from vision-language to action.

- MoE for skill learning.

    
<p style="color: purple;"> Our fundamental idea is to create an VLA model that is capable to dealing with wide range of VL-based works, to explicitly reasoning and utilize the skilles learned from VLS. Namly we call it ``VLAnything``. </p>

### Preparing Training Data

```
python label_substeps.py         --apd_path APD_plans.json         --rlds_data_dir ../LIBERO/modified_libero_rlds/         --output_path substep_labels_output.json         --suites libero_spatial_no_noops         --max_episodes 1 --debug

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_substep.py   --vla_path openvla/openvla-7b   --data_root_dir /home/aup/YuhangWorkspace/LIBERO/modified_libero_rlds   --dataset_name libero_4_task_suites_no_noops   --substep_labels_path substep_labels_output.json   --run_root_dir runs   --use_l1_regression True   --use_diffusion False   --use_film False   --num_images_in_input 2   --use_proprio True   --batch_size 1   --learning_rate 5e-4   --num_steps_before_decay 100000   --max_steps 150005   --save_freq 10000   --save_latest_checkpoint_only False   --image_aug True   --lora_rank 32   --wandb_entity "YOUR_WANDB_ENTITY"   --wandb_project "YOUR_WANDB_PROJECT"   --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```
    
    
The following shows the data formating of example episode given in DexVLA repo:
    ~~~~
![image](https://hackmd.io/_uploads/BycdQxNMbl.png)


However, their data is not open-sourced due to `company policies`. We therefore need to prepare our own dataset. We will need to glue the APD dataset and the LIBERO expert data, which they have the following formate respectively:

![image](https://hackmd.io/_uploads/B1GogmNfWx.png)



I ran experiments to provide the following statistical details of the generated APD given raw LIBERO data:

APD STATs:

| Skill                      | Substep Phrase | Occurance |
| --------------------------- | -----------:| --------------:|
| Move / Reach |         `Move gripper ..., Reach...` |   113 |
| Return to init |         `Return` |   40 |
| Pick / Grasp         |   `Pick up/Grasp/Lift` |          38 |
| Place / Put          |   `Place/Lower/Release/Move A to B` |    37 |
| Push |         `Push, close...` |            3 |
| Turn |         `Turn` |            1 |


For which, `Push` and `Turn` are rare actions, we can mark them by hand, `Return to init` appears in APD but never in action sequence since the episode terminates imidiately once the condition is fit. We only need to consider segmant timesteps into `Move`, `Pick`, and `Place`. For this, we can achive by defining a state machine:

![image](https://hackmd.io/_uploads/H1lwWQVz-x.png)

![image](https://hackmd.io/_uploads/Syp-TuUGbe.png)



The segmantation results can be found in the following:


<div style="display: flex; justify-content: center; align-items: center;">
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/HkSjp_LGbe.gif" alt="Before" width="300px" />
<figcaption></figcaption>
  </figure>
<figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/Sy9ECOUGbe.gif" alt="After" width="300px" />
    <figcaption></span></figcaption>
  </figure>
</div>!


The segmantations for each low-level action seems too short, notably, the `moving` seems not necceary, or in other words can be included into `pick` & `place` actions...



<div style="display: flex; justify-content: center; align-items: center;">
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/rko0COLfbg.gif" alt="Before" width="300px" />
<figcaption></figcaption>
  </figure>
<figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/SydbkF8zWe.gif" alt="After" width="300px" />
    <figcaption></span></figcaption>
  </figure>
</div>!


### Trail Training Results

![image](https://hackmd.io/_uploads/HkEUjwLm-g.png)



We have observe that data is largely not balanced in mordern VLA, as an example, in LIBERO, number of unique observation/action is **10000x** more than the instruction, with our efford APD, this can be reduced to **1000x**, we wanna see how this effects the model.


The most intuitive idea is to training the original openvla model by replacing the original instruction: 


![image](https://hackmd.io/_uploads/ryZYfPUQbx.png)



given the assumption that solving the data inbalance issue would be helpful.



![image](https://hackmd.io/_uploads/B1PHxwLQbg.png)
（Trained on 8x AMD MI325 for 24 hrs.）




## Attention


<div style=" justify-content: center; align-items: center;">
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/ByfA_Wt7-x.png" alt="After" width="300px" />
    <figcaption>Before T=0</figcaption></figcaption>
    </figure>
    <figure style="margin: 0 10px; text-align: center;">
        <img src="https://hackmd.io/_uploads/BkWxKbKQbg.png" alt="After" width="300px" />
    <figcaption>Before T=56</figcaption>
        <img src="https://hackmd.io/_uploads/HyvDIHKhlg.png" alt="After" width="300px" />
    <figcaption>Empty Prompt</figcaption>
  </figure>
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/rkWlwbYQWe.png" alt="After" width="300px" />
    <figcaption>Now T=0</figcaption>
  </figure>
  <figure style="margin: 0 10px; text-align: center;">
    <img src="https://hackmd.io/_uploads/HkjDwWt7-g.png" alt="After" width="300px" />
    <figcaption>Now T=56</figcaption>
  </figure>
  <figure style="margin: 0 10px; text-align: center;">
  </figure>
</div>!

### Entropy-based Evaluation



Our experiments shows that this instruction-ignorance issue is not likely a problem of the dataset prompt, we have observed serious overfits before, where the attentions of previously trained models attends only a few tokens, overfits on instructions. 


We assume that it is determined by the nature of VLA that the entropy of text is much less than the vision. 

$H(A|V) - H(A|V,L) < H(L|V)$

Only changing the propmpt increase the variaties of the instruction but does not improve the entropy of `H(L|V)`, the equation shows that the upper bound of `H(A|V) - H(A|V,L)` depends on `H(L|V)`, with a low `H(L|V)`, `H(A|V)` will be equavelent to `H(A|V,L)`. Varias prompt does not increase `H(L|V)` but APD does, hence we observe a more uniformed attention.


| Skill                      | Original | Ours |
| --------------------------- | -----------:| --------------:|
| Move / Reach |         10.2 |   67.5 |
| Pick / Grasp         |   22.4 |      52.3 |

### Advanced Benchmarks


- LIBERO-PLUS (https://arxiv.org/pdf/2510.13626)

(New training data & evaluation benckmark)

Comprehensive robustness analysis and improvement framework that systematically exposes the brittleness of current VLA models by evaluating them across seven perturbation dimensions—including camera viewpoints, lighting, and object layouts—revealing that models often rely on positional bias rather than genuine understanding. 

Beyond functioning as a diagnostic benchmark with over 10,030 tasks stratified by difficulty, it proactively addresses these failures by constructing a large-scale, automated training dataset of over 20,000 generalized trajectories.

![image](https://hackmd.io/_uploads/By6ZikZ4Zg.png)




- LIBERO-PRO (https://arxiv.org/pdf/2510.03827)

(only evaluation benckmark)

A rigorous diagnostic tool designed to expose the "rote memorization" flaw in current VLA evaluations, arguing that high scores on the standard LIBERO benchmark are artifacts of overfitting to static training conditions rather than evidence of true task comprehension. It introduces a "plug-and-play" evaluation suite that applies controlled perturbations across four dimensions—manipulated objects, initial states, instructions.

![image](https://hackmd.io/_uploads/SkEDoJbNWe.png)


## Difference in Perturbations



|                       | LIBERO-plus | LIBERO-PRO |
| --------------------------- | -----------:| --------------:|
| Sensor Apperance (Camera,Light,Sensor Noise,Robot State) |     Yes |   No |
| Task/Goal |      Yes  |   Yes |
| Language   |   Yes |    Yes |
| Objects & Layout    |   Yes (obj. pos., add. obj.) |   Yes (obj. texture, color) |
| Background/Environment |         Yes (texture) |        Yes (Environment Category) |


## APD on LIBERO-PRO


![image](https://hackmd.io/_uploads/Byq2w1Y4bl.png)

Trained on 8xMI350 for 30 hrs.

- APD Stats

![image](https://hackmd.io/_uploads/B1n4eOcNbl.png)


- Current Method

![image](https://hackmd.io/_uploads/Sk88cS5Ebl.png)


- Action Re-chunking

The original OpenVLA performs action chunking with $a_c = 8$, to avoid action chunking blindly perform actions across substeps, our method drop the remaining actions once the clip detection threshold is satisfied.


- Video

## Experiments on LIBERO-PRO

![image](https://hackmd.io/_uploads/Sk5RTTh4Wg.png)


An interesting intuitive idea is to test how libero-plus model would perform in libero-pro.


| | LIBERO-Goal | LIBERO-Goal | LIBERO-Goal | LIBERO-Goal | LIBERO-Goal | LIBERO-Spatial | LIBERO-Spatial | LIBERO-Spatial | LIBERO-Spatial | LIBERO-Spatial | LIBERO-10 | LIBERO-10 | LIBERO-10 | LIBERO-10 | LIBERO-10 | LIBERO-Object | LIBERO-Object | LIBERO-Object | LIBERO-Object | LIBERO-Object | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Obj** | **Pos** | **Sem** | **Task** | **Env** | **Obj** | **Pos** | **Sem** | **Task** | **Env** | **Obj** | **Pos** | **Sem** | **Task** | **Env** | **Obj** | **Pos** | **Sem** | **Task** | **Env** | **Total** |
| OpenVLA (per-task) | 0.96 | 0.00 | 0.98 | 0.00 | 0.98 | 0.97 | 0.00 | 0.97 | 0.00 | 0.89 | 0.81 | 0.00 | 0.96 | 0.00 | 0.85 | 0.98 | 0.00 | 0.98 | 0.00 | 0.00 | 0.52 |
| SimpleVLA (per-task) | 0.98 | 0.03 | 0.96 | 0.01 | 0.98 | 0.98 | 0.01 | -- | 
| Pi0 | 0.94 | 0.00 | 0.93 | 0.00 | 0.39 | 0.95 | 0.00 | 0.97 | 0.00 | 0.60 | 0.79 | 0.00 | 0.82 | 0.00 | 0.27 | 0.94 | 0.00 | 0.90 | 0.00 | 0.29 | 0.44 |
| Pi0.5 | 0.97 | 0.38 | 0.97 | 0.00 | 0.46 | 0.97 | 0.20 | 0.97 | 0.01 | 0.46 | 0.92 | 0.08 | 0.93 | 0.01 | 0.46 | 0.98 | 0.17 | 0.96 | 0.01 | 0.73 | 0.53 |
| Molmoact | 0.68 | 0.00 | 0.85 | 0.00 | - | 0.90 | 0.00 | 0.88 | 0.00 | - | 0.54 | 0.00 | 0.74 | 0.06 | - | 0.92 | 0.06 | 0.96 | 0.00 | - | 0.41 |
| NORA | 0.58 | 0.00 | 0.88 | 0.00 | - | 0.92 | 0.00 | 0.91 | 0.00 | - | 0.46 | 0.00 | 0.74 | 0.00 | - | 0.86 | 0.00 | 0.92 | 0.00 | - | 0.40 |
| x-VLA | 0.68 | 0.01 | 0.98 | 0.09 | - | 0.97 | 0.00 | 0.96 | 0.00 | - | 0.62 | 0.00 | 0.95 | 0.10 | - | 0.89 | 0.02 | 0.98 | 0.08 | - | 0.46 |
| **Our Experiments** | --- |
| OpenVLA-OFT | 0.13 | 0.00 | 0.47 | 0.04| -- | 0.41 | 0.10 | 0.56 | 0.00 | -- | 0.07 | 0.00 | 0.43 | 0.00 | -- | 0.74 | 0.00 | 0.94 | 0.00 | -- | 0.24  | 
| OpenVLA-OFT (Libero-plus) | 0.20 | 0.02 | 0.60 | 0.07| -- | 0.41 | 0.11 | 0.58 | 0.00 | -- | 0.14  | 0.00 | 0.43 | 0.01 | -- | 0.79 | 0.00 | 0.90 | 0.00| -- | 0.27  | 
| Ours-OpenVLA-OFT (APD) | 0.46 | 0.02 | 0.56 | 0.10 | -- | 0.57 | 0.10 | 0.56 | 0.10 | -- | 0.52  | 0.01 | 0.88 | 0.07 | -- | 0.91 | 0.00 | 0.99 | 0.01 | -- | 0.37 |
| Ours-OpenVLA-OFT (APD-EOS_pred) | 0.56 | 0.08 | 0.81 | 0.10 | -- | 0.50 | 0.14 | 0.43 | 0.02 | -- |  0.43 | 0.01 | 0.69 | 0.07 | -- | 0.73 | 0.00 | 0.80 | 0.00 | -- | 0.34 |



### Adapted EOS Ending Prediction

- 7D Action Head -> 8D Action Head

The prediction directly comes from action head predicition, with an auto-regressive objective, turns out the training is not valid, while the objective is not suitable for binary EOS flag, we also facing serious unbalanced data...

![image](https://hackmd.io/_uploads/SJp-J1cr-l.png)


|  | LIBERO-Goal | LIBERO-Goal | LIBERO-Goal | LIBERO-Goal | LIBERO-Goal | LIBERO-Object | LIBERO-Object | LIBERO-Object | LIBERO-Object | LIBERO-Object |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Obj** | **Pos** | **Sem** | **Task** | **Env** | **Obj** | **Pos** | **Sem** | **Task** | **Env** |
| APD_clip | 0.46 | 0.02 | 0.56 | 0.10 | -- | 0.91 | 0.00 | 0.99 | 0.01 | -- |
| Now | 0.49 | 0.02 | 0.58 | 0.10 | -- | 0.93 | 0.01 | 0.96 | 0.00 | -- |

- 7D Action Head + EOS Head + Downsampling


![image](https://hackmd.io/_uploads/ByYupqnS-e.png)


The next solution I have tried is to sperate EOS prediction from action head, and in order to minimze the effect of unbalanced data, I have added a downsampling components to allow updates with only balanced batch (e.g. 15x EOS=1; 20x EOS=0).

However, this solution facing several problems too, firstly, the use of downsampling means the updates of action loss and eos loss is not synced, hence the feature shifting happens, effects the eos learning. Secondly, the EOS updates has zero effects to the main VLA backbone, the model is hardly learning when to stop.

![image](https://hackmd.io/_uploads/By0zC53rWg.png)



- 7D Action Head + EOS Head + Weighted Loss


To address the limitations of previous methods regarding feature shifting and backbone adaptation, I implemented a 7D Action Head + EOS Head + Weighted Loss strategy that enables true end-to-end fine-tuning. By removing the downsampling buffer and allowing synchronized gradient flow from both heads to the VLA backbone, the model now learns manipulation and termination features simultaneously. Furthermore, the extreme 1:800 class imbalance is effectively managed through a global positive weight (e.g., 50.0) paired with **gradient clipping**, forcing the model to prioritize rare stop signals without destabilizing the training process.


$$
\begin{aligned}
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{action}} + \lambda_{\text{eos}} \cdot \mathcal{L}_{\text{eos}} \\
\\
\mathcal{L}_{\text{action}} &= \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{a}_i - \hat{\mathbf{a}}_i \|_1 \\
\\
\mathcal{L}_{\text{eos}} &= - \frac{1}{N} \sum_{i=1}^{N} \left[ w_{\text{pos}} \cdot y_i \cdot \log(\sigma(\hat{z}_i)) + (1 - y_i) \cdot \log(1 - \sigma(\hat{z}_i)) \right]
\end{aligned}
$$

**Where:**
* $N$: Batch size $\times$ Chunk length
* $\mathbf{a}_i, \hat{\mathbf{a}}_i$: Ground truth and predicted action vectors
* $y_i \in \{0, 1\}$: Ground truth EOS label
* $\hat{z}_i$: Predicted EOS **logit** (before activation)
* $\sigma(z) = \frac{1}{1 + e^{-z}}$: Sigmoid activation
* $w_{\text{pos}}$: Global positive weight (e.g., 50.0)

![image](https://hackmd.io/_uploads/BJpDxbCH-e.png)
