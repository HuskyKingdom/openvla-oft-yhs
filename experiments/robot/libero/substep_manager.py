"""
SubstepManager: Manages instruction decomposition and substep switching during evaluation.

This module provides functionality to:
1. Decompose high-level instructions into substeps using LLM (Qwen3-8B)
2. Judge substep completion using SigCLIP vision-language matching
3. Manage substep state transitions during episode execution
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SubstepManager:
    """
    Manages substep decomposition and switching during robot task execution.
    
    Uses LLM to decompose instructions and SigCLIP to judge substep completion.
    """
    
    def __init__(
        self,
        task_description: str,
        llm_model,
        llm_tokenizer,
        sigclip_model,
        sigclip_processor,
        completion_threshold: float,
        device: torch.device,
    ):
        """
        Initialize SubstepManager.
        
        Args:
            task_description: High-level task instruction
            llm_model: Qwen3-8B model for instruction decomposition
            llm_tokenizer: Tokenizer for LLM
            sigclip_model: SigCLIP model for vision-language matching
            sigclip_processor: Processor for SigCLIP
            completion_threshold: Similarity threshold for substep completion (0-1)
            device: Torch device for model inference
        """
        self.task_description = task_description
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.sigclip_model = sigclip_model
        self.sigclip_processor = sigclip_processor
        self.completion_threshold = completion_threshold
        self.device = device
        
        # State variables
        self.substeps: List[Dict] = []
        self.current_substep_idx: int = 0
        self.substep_switch_count: int = 0
        self.text_embeddings: Optional[torch.Tensor] = None
        
        # Decompose instruction into substeps
        self._decompose_instruction()
        
        # Precompute text embeddings for efficiency
        if len(self.substeps) > 0:
            self._precompute_text_embeddings()
    
    def _build_prompt(self) -> str:
        """
        Build prompt for LLM instruction decomposition.
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert robotic task planner. Your job is to break down a robot task into simple, sequential steps.

CRITICAL REQUIREMENTS:
1. You MUST output a valid JSON array with at least 3 steps
2. Each step must be a simple, low-level action that can be done in one motion
3. The final step must ALWAYS be: "Return to initial position"
4. Output ONLY the JSON array - no explanations, no extra text
5. Each JSON object must have exactly these three fields:
   - "step": integer (1, 2, 3, ...)
   - "subgoal": brief action description (what to do)
   - "expected_effect": observable result (what changes visually)

EXAMPLE OUTPUT FORMAT:
[
  {{"step": 1, "subgoal": "Move gripper above the black bowl", "expected_effect": "gripper positioned above the black bowl"}},
  {{"step": 2, "subgoal": "Lower gripper to grasp the bowl", "expected_effect": "gripper touching the black bowl"}},
  {{"step": 3, "subgoal": "Close gripper to pick up the bowl", "expected_effect": "robot holds the black bowl securely"}},
  {{"step": 4, "subgoal": "Move gripper above the plate", "expected_effect": "gripper positioned above the plate"}},
  {{"step": 5, "subgoal": "Lower and release the bowl on the plate", "expected_effect": "black bowl stably placed on the plate"}},
  {{"step": 6, "subgoal": "Return to initial position", "expected_effect": "robot back at neutral home pose"}}
]

TASK INSTRUCTION: {self.task_description}

Generate the step-by-step plan as a JSON array:"""
        return prompt
    
    def _decompose_instruction(self) -> None:
        """
        Decompose instruction into substeps using LLM.
        
        Populates self.substeps with parsed substep list.
        Handles errors gracefully by falling back to empty list.
        """
        try:
            prompt = self._build_prompt()
            
            # Prepare input for LLM
            messages = [
                {"role": "system", "content": "You are a helpful robotic task planning assistant."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate substep plan with deterministic settings for stability
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    temperature=0.1,  # Low temperature for more deterministic output
                    top_p=0.95,
                    do_sample=False,  # Use greedy decoding for consistency
                )
            
            # Decode output
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse JSON output
            self.substeps = self._parse_llm_output(output_text)
            
            # Validate minimum substep count
            if len(self.substeps) < 2:
                logger.warning(
                    f"[SubstepManager] Only {len(self.substeps)} substeps generated (expected ≥2). "
                    "This may indicate LLM output quality issue. Will use original instruction."
                )
                self.substeps = []  # Clear to trigger fallback to original instruction
            else:
                logger.info(f"[SubstepManager] Successfully decomposed into {len(self.substeps)} substeps")
            
        except Exception as e:
            logger.error(f"[SubstepManager] Failed to decompose instruction: {e}")
            self.substeps = []
    
    def _parse_llm_output(self, output_text: str) -> List[Dict]:
        """
        Parse LLM output to extract substep list.
        
        Args:
            output_text: Raw text output from LLM
            
        Returns:
            List of substep dictionaries
            
        Raises:
            ValueError if parsing fails
        """
        try:
            # Try to find JSON array in the output
            start_idx = output_text.find('[')
            end_idx = output_text.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON array found in output")
            
            json_str = output_text[start_idx:end_idx+1]
            
            # Fix common JSON formatting issues
            # Remove trailing commas before closing brackets/braces
            import re
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing comma before }
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing comma before ]
            
            substeps = json.loads(json_str)
            
            # Validate structure
            if not isinstance(substeps, list):
                raise ValueError("Output is not a list")
            
            # Filter and validate substeps
            valid_substeps = []
            for substep in substeps:
                if all(key in substep for key in ['step', 'subgoal', 'expected_effect']):
                    valid_substeps.append(substep)
                else:
                    logger.warning(f"[SubstepManager] Skipping invalid substep: {substep}")
            
            if len(valid_substeps) == 0:
                logger.warning("[SubstepManager] No valid substeps found in LLM output")
            
            return valid_substeps
            
        except Exception as e:
            logger.error(f"[SubstepManager] Failed to parse LLM output: {e}")
            logger.error(f"[SubstepManager] Raw output: {output_text[:500]}...")  # Truncate for readability
            return []
    
    def _precompute_text_embeddings(self) -> None:
        """
        Precompute SigCLIP text embeddings for all expected_effects.
        
        This avoids redundant text encoding during substep switching checks.
        """
        try:
            expected_effects = [substep['expected_effect'] for substep in self.substeps]
            
            # Process texts with SigCLIP
            with torch.no_grad():
                inputs = self.sigclip_processor(
                    text=expected_effects,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                text_outputs = self.sigclip_model.get_text_features(**inputs)
                # Normalize embeddings
                self.text_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
            
            logger.info(f"[SubstepManager] Precomputed text embeddings: {self.text_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"[SubstepManager] Failed to precompute text embeddings: {e}")
            self.text_embeddings = None
    
    def _compute_similarity(self, image: np.ndarray) -> float:
        """
        Compute similarity between image and current expected_effect.
        
        Args:
            image: RGB image as numpy array (H, W, 3) with dtype uint8
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if self.text_embeddings is None:
                logger.warning("[SubstepManager] Text embeddings not precomputed, returning 0.0")
                return 0.0
            
            if self.current_substep_idx >= len(self.substeps):
                return 1.0  # All substeps completed
            
            # Validate image format
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                logger.error(f"[SubstepManager] Invalid image format: type={type(image)}, dtype={getattr(image, 'dtype', 'N/A')}")
                return 0.0
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Resize to 224x224 (SigCLIP expected size)
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Process image with SigCLIP
            with torch.no_grad():
                inputs = self.sigclip_processor(
                    images=pil_image,
                    return_tensors="pt"
                ).to(self.device)
                
                image_outputs = self.sigclip_model.get_image_features(**inputs)
                # Normalize embeddings
                image_embeds = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity with current expected_effect
            current_text_embed = self.text_embeddings[self.current_substep_idx].unsqueeze(0)
            similarity = torch.cosine_similarity(image_embeds, current_text_embed, dim=-1)
            
            sim_score = float(similarity.item())
            
            # Sanity check the similarity score
            if not (0.0 <= sim_score <= 1.0):
                logger.warning(f"[SubstepManager] Unusual similarity score: {sim_score}, clamping to [0,1]")
                sim_score = max(0.0, min(1.0, sim_score))
            
            return sim_score
            
        except Exception as e:
            logger.error(f"[SubstepManager] Failed to compute similarity: {e}", exc_info=True)
            return 0.0
    
    def should_switch_substep(self, observation_image: np.ndarray) -> bool:
        """
        Check if current substep is completed and should switch to next.
        
        Args:
            observation_image: Current observation RGB image (H, W, 3)
            
        Returns:
            True if should switch to next substep, False otherwise
        """
        # If no substeps or already at the end, don't switch
        if len(self.substeps) == 0 or self.current_substep_idx >= len(self.substeps):
            return False
        
        # Compute similarity between observation and expected_effect
        similarity = self._compute_similarity(observation_image)
        
        # Log similarity for monitoring and debugging
        current_substep = self.substeps[self.current_substep_idx]
        logger.info(
            f"[SubstepManager] Substep {self.current_substep_idx+1}/{len(self.substeps)} "
            f"'{current_substep['subgoal'][:40]}...': "
            f"similarity={similarity:.3f}, threshold={self.completion_threshold:.3f}"
        )
        
        # Switch if similarity exceeds threshold
        should_switch = similarity >= self.completion_threshold
        if should_switch:
            logger.info(f"[SubstepManager] ✓ Substep completion detected! (similarity={similarity:.3f})")
        
        return should_switch
    
    def advance_substep(self) -> None:
        """
        Manually advance to next substep.
        
        Called after should_switch_substep returns True.
        """
        if self.current_substep_idx < len(self.substeps):
            self.current_substep_idx += 1
            self.substep_switch_count += 1
            
            logger.info(
                f"[SubstepManager] Advanced to substep {self.current_substep_idx+1}/{len(self.substeps)}"
            )
    
    def get_current_instruction(self) -> str:
        """
        Get the instruction for current substep.
        
        Returns:
            Current substep's subgoal, or original task_description if all substeps done
        """
        if len(self.substeps) == 0:
            # No substeps available, use original description
            return self.task_description
        
        if self.current_substep_idx >= len(self.substeps):
            # All substeps completed, use original description
            return self.task_description
        
        # Return current substep's subgoal
        return self.substeps[self.current_substep_idx]['subgoal']
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information for logging.
        
        Returns:
            Dictionary with progress information
        """
        return {
            'current_idx': self.current_substep_idx,
            'total': len(self.substeps),
            'current_subgoal': self.get_current_instruction(),
            'switches': self.substep_switch_count,
        }
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """
        Get final statistics after episode completion.
        
        Returns:
            Dictionary with final statistics
        """
        return {
            'total_substeps': len(self.substeps),
            'completed_substeps': self.current_substep_idx,
            'total_switches': self.substep_switch_count,
            'completion_rate': self.current_substep_idx / len(self.substeps) if len(self.substeps) > 0 else 0.0,
        }

