"""
Pure functional LLM models and types
"""
from pydantic_ai import StructuredDict

from pydantic import BaseModel, Field, root_validator, validator, conlist
from typing import List, Literal, Optional

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Automatically extracted metadata about the content"""
    subject_name: str = Field(
        ...,
        description="The main subject/domain (e.g., Mathematics, Physics, Computer Science, History, etc.)"
    )
    chapter_name: str = Field(
        ...,
        description="The chapter or topic name from the content (e.g., Neural Networks, Calculus, World War II, etc.)"
    )
    concept_name: str = Field(
        ...,
        description="The specific concept being studied (e.g., Perceptron, Derivatives, Treaty of Versailles, etc.)"
    )
    difficulty_level: Literal["Easy", "Medium", "Hard"] = Field(
        ...,
        description="Assessed difficulty level based on content complexity"
    )
    estimated_study_time: Optional[str] = Field(
        None,
        description="Estimated time needed to complete all materials (e.g., '2-3 hours', '45 minutes')"
    )


class Flashcard(BaseModel):
    """Individual flashcard for studying"""
    question: str = Field(
        ...,
        description="Front side of the flashcard: a concise, focused question."
    )
    answer: str = Field(
        ...,
        description="Back side of the flashcard: a clear and short answer."
    )
    key_concepts: Optional[str] = Field(
        None,
        description="Topic or key concept covered in this flashcard."
    )
    key_concepts_data: Optional[str] = Field(
        None,
        description="Detailed information about the concept to reinforce understanding."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ...,
        description="Difficulty level of the flashcard."
    )


class QuizQuestion(BaseModel):
    """Individual quiz question"""
    question: str = Field(
        ...,
        description="The quiz question."
    )
    options: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Multiple-choice options for the question (3-5 options)."
    )
    correct_answer: str = Field(
        ...,
        description="The correct answer from the provided options."
    )
    explanation: str = Field(
        ...,
        description="Brief explanation for why the correct answer is right."
    )


class Mapping(BaseModel):
    """Individual mapping for match the following exercise"""
    A: str = Field(
        ...,
        description="Item from column A."
    )
    B: str = Field(
        ...,
        description="Correctly matched item from column B."
    )


class MatchTheFollowing(BaseModel):
    """Match the following exercise structure"""
    columnA: List[str] = Field(
        ...,
        description="List of terms, definitions, or entities in column A."
    )
    columnB: List[str] = Field(
        ...,
        description="List of corresponding matches for items in column A."
    )
    mappings: List[Mapping] = Field(
        ...,
        description="Array of correct pairings between column A and column B."
    )


class LearningContent(BaseModel):
    """Main schema for learning content generation"""
    
    # Always included
    metadata: Metadata = Field(
        ...,
        description="Automatically extracted metadata about the content"
    )
    
    content_type: Literal["flashcards", "quiz", "match_the_following", "summary", "all"] = Field(
        ...,
        description="Type of content requested by the user"
    )
    
    summary: str = Field(
        ...,
        description="Comprehensive, concise summary of all key concepts covered."
    )
    
    learning_objectives: List[str] = Field(
        ...,
        description="List of learning objectives for the given topic."
    )
    
    # Optional content based on user request
    flashcards: Optional[List[Flashcard]] = Field(
        None,
        min_length=15,
        max_length=15,
        description="A set of exactly 15 flashcards summarizing important key concepts. Only include if user requests flashcards."
    )
    
    quiz: Optional[List[QuizQuestion]] = Field(
        None,
        min_length=10,
        max_length=10,
        description="A set of exactly 10 quiz questions for practice. Only include if user requests quiz."
    )
    
    match_the_following: Optional[MatchTheFollowing] = Field(
        None,
        description="A 'match the following' exercise with two columns (A and B) and correct mappings. Only include if user requests this."
    )


# Example usage with PydanticAI
"""
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4',
    result_type=LearningContent,
    system_prompt="Generate educational content based on the provided material..."
)

# Use the agent
result = await agent.run("Create flashcards for quantum mechanics")
learning_content = result.data  # This will be a LearningContent instance
"""

LEARNING_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        # Metadata extraction (always included)
        "metadata": {
            "type": "object",
            "description": "Automatically extracted metadata about the content",
            "properties": {
                "subject_name": {
                    "type": "string",
                    "description": "The main subject/domain (e.g., Mathematics, Physics, Computer Science, History, etc.)"
                },
                "chapter_name": {
                    "type": "string",
                    "description": "The chapter or topic name from the content (e.g., Neural Networks, Calculus, World War II, etc.)"
                },
                "concept_name": {
                    "type": "string",
                    "description": "The specific concept being studied (e.g., Perceptron, Derivatives, Treaty of Versailles, etc.)"
                },
                "difficulty_level": {
                    "type": "string",
                    "enum": ["Easy", "Medium", "Hard"],
                    "description": "Assessed difficulty level based on content complexity"
                },
                "estimated_study_time": {
                    "type": "string",
                    "description": "Estimated time needed to complete all materials (e.g., '2-3 hours', '45 minutes')"
                }
            },
            "required": ["subject_name", "chapter_name", "concept_name", "difficulty_level"]
        },

        # Content type requested by user
        "content_type": {
            "type": "string",
            "enum": ["flashcards", "quiz", "match_the_following", "summary", "all"],
            "description": "Type of content requested by the user"
        },

        # 15 flashcards (only when requested)
        "flashcards": {
            "type": "array",
            "description": "A set of exactly 15 flashcards summarizing important key concepts. Only include if user requests flashcards.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Front side of the flashcard: a concise, focused question."
                    },
                    "answer": {
                        "type": "string",
                        "description": "Back side of the flashcard: a clear and short answer."
                    },
                    "key_concepts": {
                        "type": "string",
                        "description": "Topic or key concept covered in this flashcard."
                    },
                    "key_concepts_data": {
                        "type": "string",
                        "description": "Detailed information about the concept to reinforce understanding."
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["Easy", "Medium", "Hard"],
                        "description": "Difficulty level of the flashcard."
                    }
                },
                "required": ["question", "answer", "difficulty"]
            }
        },

        # 10 quiz questions (only when requested)
        "quiz": {
            "type": "array",
            "description": "A set of exactly 10 quiz questions for practice. Only include if user requests quiz.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The quiz question."
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple-choice options for the question (3-5 options)."
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "The correct answer from the provided options."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for why the correct answer is right."
                    }
                },
                "required": ["question", "options", "correct_answer", "explanation"]
            }
        },

        # Match the following section (only when requested)
        "match_the_following": {
            "type": "object",
            "description": "A 'match the following' exercise with two columns (A and B) and correct mappings. Only include if user requests this.",
            "properties": {
                "columnA": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of terms, definitions, or entities in column A."
                },
                "columnB": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of corresponding matches for items in column A."
                },
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "A": {
                                "type": "string",
                                "description": "Item from column A."
                            },
                            "B": {
                                "type": "string",
                                "description": "Correctly matched item from column B."
                            }
                        },
                        "required": ["A", "B"]
                    },
                    "description": "Array of correct pairings between column A and column B."
                }
            },
            "required": ["columnA", "columnB", "mappings"]
        },

        # Summary (always included)
        "summary": {
            "type": "string",
            "description": "Comprehensive, concise summary of all key concepts covered."
        },

        # Learning objectives (always included)
        "learning_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of learning objectives for the given topic."
        }
    },
    "required": ["metadata", "content_type", "summary", "learning_objectives"]
}


  
LEARNING_CONTENT = StructuredDict(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metadata": {
                "type": "object",
                "description": "Automatically extracted metadata about the content",
                "additionalProperties": False,
                "properties": {
                    "subject_name": {"type": "string"},
                    "chapter_name": {"type": "string"},
                    "concept_name": {"type": "string"},
                    "difficulty_level": {
                        "type": "string",
                        "enum": ["Easy", "Medium", "Hard"]
                    },
                    "estimated_study_time": {"type": "string"}
                },
                "required": ["subject_name", "chapter_name", "concept_name", "difficulty_level"]
            },
            "content_type": {
                "type": "string",
                "enum": ["flashcards", "quiz", "match_the_following", "summary", "all"],
                "description": "Requested content type. 'all' means include flashcards, quiz and match_the_following."
            },
            "flashcards": {
                "type": "array",
                "description": "When present, should contain up to 10 flashcards.",
                "minItems": 0,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "answer": {"type": "string", "minLength": 1},
                        "key_concepts": {"type": "array", "items": {"type": "string"}},
                        "key_concepts_data": {"type": "string"},
                        "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]}
                    },
                    "required": ["question", "answer", "difficulty"]
                }
            },
            "quiz": {
                "type": "array",
                "description": "When present, contains multiple-choice questions (enforce 10 items).",
                "minItems": 0,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 4
                        },
                        "correct_answer": {"type": "string"},
                        "explanation": {"type": "string"},
                        "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]}
                    },
                    "required": ["question", "options", "correct_answer", "explanation"]
                }
            },
            "match_the_following": {
                "type": "object",
                "description": "Match exercise when present.",
                "additionalProperties": False,
                "properties": {
                    "columnA": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
                    "columnB": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
                    "mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"A": {"type": "string"}, "B": {"type": "string"}},
                            "required": ["A", "B"]
                        },
                        "maxItems": 8
                    }
                }
            },
            "summary": {"type": "string", "minLength": 10},
            "learning_objectives": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}
        },
        "required": ["metadata", "content_type", "summary", "learning_objectives"]
    }
)