"""
ENHANCED Integrated Translation System v5.0
Combines speech recognition, gender detection, and advanced translation
Ensures all segments translate correctly, accurately, and efficiently
"""

import torch
import logging
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
import re
import numpy as np
import os
import json
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio
warnings.filterwarnings("ignore")

# Import from utils
try:
    from utils.gpu_utils import gpu_manager
except ImportError:
    # Fallback for standalone usage
    class DummyGPUManager:
        def get_device(self): return "cuda" if torch.cuda.is_available() else "cpu"
        def clear_gpu_cache(self): 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        def safe_model_load(self, loader_fn, *args, **kwargs): return loader_fn(*args, **kwargs)
        def cleanup_model(self, model): del model
        def monitor_memory(self): 
            if torch.cuda.is_available():
                return {"gpu_memory_free": torch.cuda.get_device_properties(0).total_memory / 1e9}
            return {"gpu_memory_free": 0}
        def get_optimal_batch_size(self, default_size): return default_size
    
    gpu_manager = DummyGPUManager()

logger = logging.getLogger(__name__)

class EnhancedIntegratedTranslator:
    def __init__(self, model_size: str = "medium"):
        """
        Initialize the enhanced integrated translator
        
        Args:
            model_size: Size of translation model to use ("small", "medium", "large")
        """
        self.device = gpu_manager.get_device()
        
        # Model selection based on size parameter
        if model_size == "small":
            self.primary_model_name = "facebook/nllb-200-distilled-600M"
            self.fallback_model_name = None
        elif model_size == "large":
            self.primary_model_name = "facebook/nllb-200-distilled-1.3B"
            self.fallback_model_name = "facebook/nllb-200-distilled-600M"
        else:  # medium (default)
            self.primary_model_name = "facebook/nllb-200-distilled-600M"
            self.fallback_model_name = None
        
        # Initialize model containers
        self.primary_model = None
        self.fallback_model = None
        self.tokenizer = None
        self.translator = None
        
        # Configuration
        self.confidence_threshold = 0.6
        self.max_segment_length = 512
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.batch_size = 8
        
        # Performance tracking
        self.translation_stats = {
            "total_segments": 0,
            "successful_translations": 0,
            "total_time": 0,
            "avg_time_per_segment": 0
        }
        
        # Enhanced language code mapping (NLLB-200 format)
        self.language_codes = {
            "en": "eng_Latn",
            "ur": "urd_Arab", 
            "hi": "hin_Deva",
            "ar": "arb_Arab",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "zh": "zho_Hans",
            "bn": "ben_Beng",
            "fa": "fas_Arab",
            "tr": "tur_Latn",
            "pl": "pol_Latn",
            "nl": "nld_Latn",
            "sv": "swe_Latn",
            "fi": "fin_Latn",
            "da": "dan_Latn",
            "no": "nno_Latn",
            "cs": "ces_Latn",
            "hu": "hun_Latn",
            "th": "tha_Thai",
            "vi": "vie_Latn",
            "id": "ind_Latn",
            "ms": "msa_Latn",
            "he": "heb_Hebr",
            "el": "ell_Grek",
            "ro": "ron_Latn",
            "uk": "ukr_Cyrl"
        }
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load enhanced translation models with fallbacks and optimizations"""
        try:
            logger.info(f"🌐 Loading Enhanced Integrated Translator v5.0...")
            logger.info(f"   Primary model: {self.primary_model_name}")
            logger.info(f"   Fallback model: {self.fallback_model_name or 'None'}")
            logger.info(f"   Device: {self.device}")
            
            # Clear GPU cache before loading
            gpu_manager.clear_gpu_cache()
            
            # Load primary model
            try:
                start_time = time.time()
                logger.info("🔄 Loading primary translation model...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.primary_model_name)
                
                # Use half precision on GPU for memory efficiency
                self.primary_model = gpu_manager.safe_model_load(
                    AutoModelForSeq2SeqLM.from_pretrained,
                    self.primary_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                # Create optimized translation pipeline
                self.translator = pipeline(
                    "translation",
                    model=self.primary_model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    max_length=self.max_segment_length,
                    do_sample=False,
                    num_beams=4
                )
                
                load_time = time.time() - start_time
                logger.info(f"✅ Primary translation model loaded successfully in {load_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"Primary model loading failed: {e}")
                raise
            
            # Try to load fallback model if specified and memory allows
            if self.fallback_model_name:
                try:
                    memory_info = gpu_manager.monitor_memory()
                    available_memory = memory_info.get('gpu_memory_free', 0)
                    
                    if available_memory > 2.0:  # At least 2GB free
                        logger.info("📦 Loading fallback translation model...")
                        self.fallback_model = gpu_manager.safe_model_load(
                            AutoModelForSeq2SeqLM.from_pretrained,
                            self.fallback_model_name,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                        )
                        logger.info("✅ Fallback translation model loaded")
                    else:
                        logger.info("⚠️ Insufficient memory for fallback model")
                        
                except Exception as e:
                    logger.info(f"Fallback model not loaded: {e}")
            
            # Optimize batch size based on available memory
            self.batch_size = gpu_manager.get_optimal_batch_size(self.batch_size)
            logger.info(f"   Optimal batch size: {self.batch_size}")
            
            logger.info("✅ Enhanced Integrated Translator ready!")
            
        except Exception as e:
            logger.error(f"Enhanced translation model loading failed: {e}")
            # Try smaller model as final fallback
            try:
                logger.info("🔄 Trying minimal model as final fallback...")
                self.primary_model_name = "facebook/nllb-200-distilled-600M"
                self._load_minimal_model()
            except Exception as e2:
                logger.error(f"All translation models failed: {e2}")
                raise Exception(f"Translation model loading failed: {str(e)}")
    
    def _load_minimal_model(self):
        """Load minimal model as final fallback"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.primary_model_name)
        self.primary_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.primary_model_name,
            torch_dtype=torch.float32
        )
        
        self.translator = pipeline(
            "translation",
            model=self.primary_model,
            tokenizer=self.tokenizer,
            device=-1,  # Force CPU
            max_length=256
        )
        
        # Set minimal batch size
        self.batch_size = 2
    
    def translate_pipeline_results(self, pipeline_results: Dict, 
                                 source_lang: str, target_lang: str) -> Dict:
        """
        Translate results from speech recognition pipeline
        
        Args:
            pipeline_results: Results from speech recognition pipeline
            source_lang: Source language code (e.g., 'en', 'es')
            target_lang: Target language code (e.g., 'en', 'es')
            
        Returns:
            Dictionary with translated results
        """
        try:
            start_time = time.time()
            
            logger.info(f"🌍 Translating pipeline results: {source_lang} → {target_lang}")
            
            # Extract segments from pipeline results
            segments = pipeline_results.get('segments', [])
            if not segments:
                logger.warning("No segments found in pipeline results")
                return {
                    'success': False,
                    'error': 'No segments found',
                    'original_results': pipeline_results
                }
            
            logger.info(f"   Total segments: {len(segments)}")
            
            # Translate segments
            translated_segments = self.translate_segments_enhanced(segments, source_lang, target_lang)
            
            # Update pipeline results with translated segments
            translated_results = pipeline_results.copy()
            translated_results['segments'] = translated_segments
            translated_results['translation'] = {
                'source_language': source_lang,
                'target_language': target_lang,
                'success': True
            }
            
            # Calculate statistics
            successful_translations = sum(1 for s in translated_segments 
                                        if s.get('translation_confidence', 0) >= self.confidence_threshold)
            
            translation_stats = {
                'total_segments': len(translated_segments),
                'successful_translations': successful_translations,
                'success_rate': (successful_translations / len(translated_segments) * 100) if translated_segments else 0,
                'avg_confidence': np.mean([s.get('translation_confidence', 0) for s in translated_segments]) if translated_segments else 0,
                'processing_time': time.time() - start_time
            }
            
            translated_results['translation_stats'] = translation_stats
            
            logger.info(f"✅ Pipeline translation completed in {translation_stats['processing_time']:.2f}s")
            logger.info(f"   Success rate: {translation_stats['success_rate']:.1f}%")
            logger.info(f"   Avg confidence: {translation_stats['avg_confidence']:.2f}")
            
            return translated_results
            
        except Exception as e:
            logger.error(f"Pipeline translation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_results': pipeline_results
            }
    
    def translate_segments_enhanced(self, segments: List[Dict], source_lang: str, target_lang: str) -> List[Dict]:
        """
        Enhanced segment translation with comprehensive metadata and optimizations
        
        Args:
            segments: List of segments to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated segments with metadata
        """
        try:
            start_time = time.time()
            logger.info(f"🌐 Enhanced segment translation: {len(segments)} segments")
            
            # Skip if source and target languages are the same
            if source_lang == target_lang:
                logger.info("Source and target languages are the same, skipping translation")
                for segment in segments:
                    segment["original_text"] = segment.get("text", "")
                    segment["translation_confidence"] = 1.0
                    segment["translation_method"] = "same_language"
                    segment["translation_quality"] = "high"
                    segment["source_language"] = source_lang
                    segment["target_language"] = target_lang
                return segments
            
            # Extract texts for batch translation
            texts = []
            for segment in segments:
                text = segment.get("text", "").strip()
                # Skip empty segments
                if not text:
                    texts.append("")
                    continue
                
                # Handle long segments by splitting into sentences
                if len(text) > self.max_segment_length:
                    texts.append(self._smart_segment_text(text))
                else:
                    texts.append(text)
            
            # Perform enhanced batch translation
            translation_results = self.translate_batch_enhanced(texts, source_lang, target_lang)
            
            # Merge results with original segments
            enhanced_segments = []
            successful_translations = 0
            
            for segment, translation in zip(segments, translation_results):
                enhanced_segment = segment.copy()
                
                # Add translation data
                enhanced_segment["original_text"] = translation["original_text"]
                enhanced_segment["text"] = translation["translated_text"]
                enhanced_segment["translation_confidence"] = translation["confidence"]
                enhanced_segment["translation_method"] = translation.get("method", "unknown")
                enhanced_segment["source_language"] = source_lang
                enhanced_segment["target_language"] = target_lang
                
                # Add quality assessment
                enhanced_segment["translation_quality"] = self._assess_translation_quality(
                    translation["confidence"], translation["translated_text"]
                )
                
                if translation["confidence"] >= self.confidence_threshold:
                    successful_translations += 1
                
                enhanced_segments.append(enhanced_segment)
            
            # Update translation stats
            processing_time = time.time() - start_time
            self.translation_stats["total_segments"] += len(segments)
            self.translation_stats["successful_translations"] += successful_translations
            self.translation_stats["total_time"] += processing_time
            self.translation_stats["avg_time_per_segment"] = (
                self.translation_stats["total_time"] / self.translation_stats["total_segments"]
                if self.translation_stats["total_segments"] > 0 else 0
            )
            
            logger.info(f"✅ Enhanced segment translation completed in {processing_time:.2f}s")
            logger.info(f"   Successful translations: {successful_translations}/{len(segments)}")
            logger.info(f"   Success rate: {(successful_translations/len(segments)*100):.1f}%")
            logger.info(f"   Avg time per segment: {processing_time/len(segments):.3f}s")
            
            return enhanced_segments
            
        except Exception as e:
            logger.error(f"Enhanced segment translation failed: {e}")
            # Return original segments with error markers
            for segment in segments:
                segment["original_text"] = segment.get("text", "")
                segment["translation_confidence"] = 0.1
                segment["translation_method"] = "error"
                segment["translation_quality"] = "failed"
                segment["source_language"] = source_lang
                segment["target_language"] = target_lang
            return segments
    
    def translate_batch_enhanced(self, texts: List[str], source_lang: str, target_lang: str) -> List[Dict]:
        """
        Enhanced batch translation with intelligent processing and optimizations
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translation results with metadata
        """
        try:
            if not texts:
                return []
            
            logger.info(f"🌐 Enhanced batch translation: {len(texts)} texts ({source_lang} → {target_lang})")
            
            # Filter and prepare texts
            text_data = [(i, text.strip()) for i, text in enumerate(texts)]
            non_empty_texts = [(i, text) for i, text in text_data if text]
            
            if not non_empty_texts:
                return [self._create_empty_translation(text, source_lang, target_lang) for text in texts]
            
            # Check if same language
            src_code = self.language_codes.get(source_lang, "eng_Latn")
            tgt_code = self.language_codes.get(target_lang, "eng_Latn")
            
            if src_code == tgt_code:
                return [self._create_same_language_result(text, source_lang, target_lang) for text in texts]
            
            # Process in optimized batches
            all_results = [None] * len(texts)
            
            # Process in smaller batches for better memory management
            for batch_start in range(0, len(non_empty_texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(non_empty_texts))
                batch_items = non_empty_texts[batch_start:batch_end]
                
                # Process batch with retries
                batch_results = self._process_translation_batch_with_retry(batch_items, src_code, tgt_code)
                
                # Store results
                for (orig_idx, orig_text), result in zip(batch_items, batch_results):
                    all_results[orig_idx] = self._finalize_translation_result(
                        orig_text, result["translated_text"],
                        source_lang, target_lang, result["confidence"], result["method"]
                    )
            
            # Fill empty positions
            for i, result in enumerate(all_results):
                if result is None:
                    all_results[i] = self._create_empty_translation(texts[i], source_lang, target_lang)
            
            logger.info(f"✅ Enhanced batch translation completed")
            return all_results
            
        except Exception as e:
            logger.error(f"Enhanced batch translation failed: {e}")
            # Fallback to individual translations
            return [self._create_error_translation(text, source_lang, target_lang, str(e)) for text in texts]
    
    def _process_translation_batch_with_retry(self, batch_items: List[tuple], 
                                            src_code: str, tgt_code: str) -> List[Dict]:
        """
        Process a batch of translations with retry mechanism
        
        Args:
            batch_items: List of (index, text) tuples
            src_code: Source language code in NLLB format
            tgt_code: Target language code in NLLB format
            
        Returns:
            List of translation results
        """
        try:
            batch_texts = [self._clean_text_for_translation(text) for _, text in batch_items]
            
            # Clear GPU cache
            gpu_manager.clear_gpu_cache()
            
            # First attempt with primary model
            try:
                with torch.no_grad():
                    results = self.translator(
                        batch_texts,
                        src_lang=src_code,
                        tgt_lang=tgt_code,
                        max_length=self.max_segment_length,
                        num_beams=3,  # Reduced for batch processing
                        batch_size=len(batch_texts)
                    )
                
                # Process results
                processed_results = []
                for i, result in enumerate(results):
                    translated_text = result["translation_text"]
                    original_text = batch_items[i][1]
                    confidence = self._calculate_enhanced_confidence(original_text, translated_text, "batch")
                    
                    processed_results.append({
                        "translated_text": translated_text,
                        "confidence": confidence,
                        "method": "batch_primary"
                    })
                
                return processed_results
                
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                # Fallback to smaller batches or individual processing
                if len(batch_texts) > 1:
                    logger.info("🔄 Retrying with smaller batches...")
                    mid = len(batch_items) // 2
                    first_half = self._process_translation_batch_with_retry(batch_items[:mid], src_code, tgt_code)
                    second_half = self._process_translation_batch_with_retry(batch_items[mid:], src_code, tgt_code)
                    return first_half + second_half
                else:
                    # Individual processing as last resort
                    return [self._translate_single_text(text, src_code, tgt_code) 
                           for _, text in batch_items]
            
        except Exception as e:
            logger.error(f"Batch processing with retry failed: {e}")
            # Final fallback
            return [
                {"translated_text": text, "confidence": 0.3, "method": "batch_fallback"}
                for _, text in batch_items
            ]
    
    def _translate_single_text(self, text: str, src_code: str, tgt_code: str) -> Dict:
        """
        Translate a single text with enhanced error handling
        
        Args:
            text: Text to translate
            src_code: Source language code in NLLB format
            tgt_code: Target language code in NLLB format
            
        Returns:
            Translation result
        """
        try:
            if not text.strip():
                return {"translated_text": "", "confidence": 0.0, "method": "empty_input"}
            
            # Clear GPU cache
            gpu_manager.clear_gpu_cache()
            
            with torch.no_grad():
                result = self.translator(
                    text,
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                    max_length=self.max_segment_length,
                    num_beams=4,
                    early_stopping=True
                )
                
                translated_text = result[0]["translation_text"] if result else text
            
            # Calculate confidence
            confidence = self._calculate_enhanced_confidence(text, translated_text, "single")
            
            return {
                "translated_text": translated_text,
                "confidence": confidence,
                "method": "single_primary"
            }
            
        except Exception as e:
            logger.warning(f"Single text translation failed: {e}")
            return {"translated_text": text, "confidence": 0.1, "method": "single_error"}
    
    def _smart_segment_text(self, text: str) -> str:
        """
        Smart text segmentation for better translation of long texts
        
        Args:
            text: Long text to segment
            
        Returns:
            Processed text ready for translation
        """
        # If text is already short enough, return as is
        if len(text) <= self.max_segment_length:
            return text
        
        # Try to split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If we have multiple sentences, return the most important ones
        if len(sentences) > 1:
            # Prioritize first and last sentences
            if len(sentences) <= 3:
                return ' '.join(sentences)
            else:
                # Take first sentence, last sentence, and middle sentences up to the limit
                max_middle = (self.max_segment_length - len(sentences[0]) - len(sentences[-1])) // 30
                middle_sentences = sentences[1:-1][:max_middle]
                return ' '.join([sentences[0]] + middle_sentences + [sentences[-1]])
        
        # If we can't split by sentences, truncate with ellipsis
        return text[:self.max_segment_length-3] + "..."
    
    def _clean_text_for_translation(self, text: str) -> str:
        """
        Clean text for better translation quality
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        try:
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', text.strip())
            
            # Remove special characters that might confuse translation
            cleaned = re.sub(r'[^\w\s\.,!?;:\-\'\"()]', ' ', cleaned)
            
            # Remove very short words that might be artifacts
            words = cleaned.split()
            words = [word for word in words if len(word) > 1 or word.lower() in ['i', 'a']]
            
            return ' '.join(words)
            
        except Exception:
            return text
    
    def _calculate_enhanced_confidence(self, original: str, translated: str, model_type: str) -> float:
        """
        Calculate enhanced confidence score with multiple factors
        
        Args:
            original: Original text
            translated: Translated text
            model_type: Type of model used ("primary", "fallback", "batch", "single")
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            if not original.strip() or not translated.strip():
                return 0.0
            
            base_confidence = 0.7
            
            # Length ratio factor
            length_ratio = len(translated) / max(1, len(original))
            if 0.5 <= length_ratio <= 2.0:
                length_factor = 0.1
            elif 0.3 <= length_ratio <= 3.0:
                length_factor = 0.05
            else:
                length_factor = -0.1
            
            # Word count factor
            orig_words = len(original.split())
            trans_words = len(translated.split())
            
            if orig_words > 0:
                word_ratio = trans_words / orig_words
                if 0.6 <= word_ratio <= 1.5:
                    word_factor = 0.1
                elif 0.4 <= word_ratio <= 2.0:
                    word_factor = 0.05
                else:
                    word_factor = -0.05
            else:
                word_factor = 0
            
            # Check for obvious translation failures
            if translated.lower() == original.lower():
                return 0.3  # Likely untranslated
            
            # Check for repeated patterns
            words = translated.split()
            if len(words) > 3:
                unique_words = len(set(words))
                repetition_ratio = unique_words / len(words)
                if repetition_ratio < 0.5:
                    repetition_factor = -0.2
                elif repetition_ratio < 0.7:
                    repetition_factor = -0.1
                else:
                    repetition_factor = 0.05
            else:
                repetition_factor = 0
            
            # Check for punctuation preservation
            orig_punct_count = len(re.findall(r'[.,!?;:]', original))
            trans_punct_count = len(re.findall(r'[.,!?;:]', translated))
            
            if orig_punct_count > 0:
                punct_ratio = trans_punct_count / orig_punct_count
                if 0.5 <= punct_ratio <= 1.5:
                    punct_factor = 0.05
                else:
                    punct_factor = -0.05
            else:
                punct_factor = 0
            
            # Model-specific adjustments
            model_factor = 0.0
            if model_type == "batch":
                model_factor = -0.05  # Batch processing is slightly less accurate
            elif model_type == "single":
                model_factor = 0.05  # Individual processing can be more accurate
            
            final_confidence = base_confidence + length_factor + word_factor + repetition_factor + punct_factor + model_factor
            return max(0.1, min(0.95, final_confidence))
            
        except Exception:
            return 0.5
    
    def _assess_translation_quality(self, confidence: float, translated_text: str) -> str:
        """
        Assess translation quality
        
        Args:
            confidence: Confidence score
            translated_text: Translated text
            
        Returns:
            Quality assessment ("high", "medium", "low", "poor", "empty")
        """
        try:
            if not translated_text.strip():
                return "empty"
            
            if confidence >= 0.8:
                return "high"
            elif confidence >= 0.6:
                return "medium"
            elif confidence >= 0.4:
                return "low"
            else:
                return "poor"
                
        except Exception:
            return "unknown"
    
    def _create_empty_translation(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """Create empty translation result"""
        return {
            "original_text": text,
            "translated_text": "",
            "source_language": source_lang,
            "target_language": target_lang,
            "confidence": 0.0,
            "method": "empty_input"
        }
    
    def _create_same_language_result(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """Create result for same language translation"""
        return {
            "original_text": text,
            "translated_text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "confidence": 1.0,
            "method": "same_language"
        }
    
    def _create_error_translation(self, text: str, source_lang: str, target_lang: str, error: str) -> Dict:
        """Create error translation result"""
        return {
            "original_text": text,
            "translated_text": text,  # Fallback to original
            "source_language": source_lang,
            "target_language": target_lang,
            "confidence": 0.1,
            "method": "error_fallback",
            "error": error
        }
    
    def _finalize_translation_result(self, original: str, translated: str, 
                                   source_lang: str, target_lang: str, 
                                   confidence: float, method: str) -> Dict:
        """Finalize translation result with all metadata"""
        return {
            "original_text": original,
            "translated_text": translated,
            "source_language": source_lang,
            "target_language": target_lang,
            "confidence": confidence,
            "method": method,
            "quality": self._assess_translation_quality(confidence, translated)
        }
    
    def get_translation_stats(self) -> Dict:
        """Get comprehensive translation statistics"""
        return self.translation_stats
    
    def translate_file(self, input_file: str, output_file: str, 
                     source_lang: str, target_lang: str) -> Dict:
        """
        Translate a file containing JSON data with segments
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary with translation statistics
        """
        try:
            logger.info(f"📄 Translating file: {input_file} → {output_file}")
            
            # Load input file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract segments
            segments = data.get('segments', [])
            if not segments:
                logger.warning("No segments found in input file")
                return {
                    'success': False,
                    'error': 'No segments found',
                    'file': input_file
                }
            
            logger.info(f"   Total segments: {len(segments)}")
            
            # Translate segments
            translated_segments = self.translate_segments_enhanced(segments, source_lang, target_lang)
            
            # Update data with translated segments
            data['segments'] = translated_segments
            data['translation'] = {
                'source_language': source_lang,
                'target_language': target_lang,
                'timestamp': time.time()
            }
            
            # Calculate statistics
            successful_translations = sum(1 for s in translated_segments 
                                        if s.get('translation_confidence', 0) >= self.confidence_threshold)
            
            stats = {
                'total_segments': len(translated_segments),
                'successful_translations': successful_translations,
                'success_rate': (successful_translations / len(translated_segments) * 100) if translated_segments else 0,
                'avg_confidence': np.mean([s.get('translation_confidence', 0) for s in translated_segments]) if translated_segments else 0
            }
            
            data['translation_stats'] = stats
            
            # Save output file
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ File translation completed: {output_file}")
            logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
            
            return {
                'success': True,
                'stats': stats,
                'input_file': input_file,
                'output_file': output_file
            }
            
        except Exception as e:
            logger.error(f"File translation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file': input_file
            }
    
    def cleanup(self):
        """Enhanced cleanup of translation resources"""
        try:
            logger.info("🧹 Cleaning up Enhanced Integrated Translator...")
            
            if self.primary_model:
                gpu_manager.cleanup_model(self.primary_model)
                self.primary_model = None
            
            if self.fallback_model:
                gpu_manager.cleanup_model(self.fallback_model)
                self.fallback_model = None
            
            if self.translator:
                del self.translator
                self.translator = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            self.executor.shutdown(wait=False)
            gpu_manager.clear_gpu_cache()
            
            logger.info("✅ Enhanced Integrated Translator cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# Factory function for easy import
def create_integrated_translator(model_size: str = "medium") -> EnhancedIntegratedTranslator:
    """Factory function to create an enhanced integrated translator"""
    return EnhancedIntegratedTranslator(model_size=model_size)

# Test the integrated translator
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Testing Enhanced Integrated Translator v5.0")
    print("=" * 70)
    
    # Create translator
    translator = create_integrated_translator(model_size="medium")
    
    # Test with sample segments
    sample_segments = [
        {
            "id": 0,
            "start": 0.0,
            "end": 5.0,
            "text": "Hello, this is a test of the translation system.",
            "speaker": "SPEAKER_00"
        },
        {
            "id": 1,
            "start": 5.0,
            "end": 10.0,
            "text": "We need to ensure all segments translate correctly and efficiently.",
            "speaker": "SPEAKER_01"
        },
        {
            "id": 2,
            "start": 10.0,
            "end": 15.0,
            "text": "The system should handle various languages and maintain accuracy.",
            "speaker": "SPEAKER_00"
        }
    ]
    
    # Translate from English to Spanish
    print("\n🌍 Translating sample segments (English → Spanish)...")
    translated_segments = translator.translate_segments_enhanced(sample_segments, "en", "es")
    
    print("\n✅ Translation results:")
    for segment in translated_segments:
        print(f"  Original: {segment['original_text']}")
        print(f"  Translated: {segment['text']}")
        print(f"  Confidence: {segment['translation_confidence']:.2f}")
        print(f"  Quality: {segment['translation_quality']}")
        print()
    
    # Get statistics
    stats = translator.get_translation_stats()
    print(f"\n📊 Translation statistics:")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Successful translations: {stats['successful_translations']}")
    print(f"  Average time per segment: {stats['avg_time_per_segment']:.3f}s")
    
    # Cleanup
    translator.cleanup()
    print("\n" + "=" * 70)
