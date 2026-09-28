import json
import uuid
import logging
from pathlib import Path
from groq import AsyncGroq
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.agent.memory import prompts
from app.models import ChatHistory, PerDocMemory
from app.config import USER_PROFILE_FILE, SOUL_FILE, CONSOLIDATE_EVERY_N

logger = logging.getLogger(__name__)



class Memory:
    def __init__(self, db: AsyncSession, client: AsyncGroq, small_model: str, agent_dir: Path): 
        self.db = db
        self.client = client
        self.small_model = small_model
        self.agent_dir = agent_dir
        self.user_profile_path = self.agent_dir / USER_PROFILE_FILE
        self.soul_path = self.agent_dir / SOUL_FILE

        self._build_local_mem()


    def _build_local_mem(self):
        """Build local memory in .agent folder including memory about user and agent personality"""
        self.agent_dir.mkdir(exist_ok=True)

        if not self.user_profile_path.exists():
            self.user_profile_path.write_text(
                "No information about this user yet."
            )
        
        if not self.soul_path.exists():
            self.soul_path.write_text(
                (
                    "You are a research assistant specialized in helping users deeply understand academic papers and documents.\n"
                    "\n"
                    "## Core behavior\n"
                    "- Answer based on the document content. If the information is not in the documents, say so clearly.\n"
                    "- Be direct and precise. Avoid unnecessary preamble or filler phrases.\n"
                    "- Match the depth of your answer to the complexity of the question — simple questions get concise answers, complex ones get thorough ones.\n"
                    "- When citing information, indicate which document or section it comes from.\n"
                    "\n"
                    "## When reading papers\n"
                    "- Help the user understand not just what the paper says, but why it matters.\n"
                    "- Point out connections between concepts when relevant.\n"
                    "- Flag assumptions, limitations, or weak points in the methodology if asked.\n"
                    "\n"
                    "## Tone\n"
                    "- Treat the user as an intelligent adult who does not need hand-holding.\n"
                    "- Be collegial, not servile — push back if the user's interpretation seems off.\n"
                ),
                encoding="utf-8",
            )


    def _get_user_profile(self) -> str:
        """Get current user profile in USER.md"""
        return self.user_profile_path.read_text(encoding='utf-8')


    async def _get_doc_memory_content(self, doc_id: uuid.UUID) -> str | None:
        """Get current memory content of user about the document"""
        try:
            doc_mem = (
                await self.db.scalar(
                    select(PerDocMemory)
                    .where(PerDocMemory.document_id == doc_id)
                )
            )

            return doc_mem.content if doc_mem else None
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e


    async def _get_doc_memory(self, doc_id: uuid.UUID) -> PerDocMemory | None:
        """Get current memory of user about the document"""
        try:
            doc_mem = (
                await self.db.scalar(
                    select(PerDocMemory)
                    .where(PerDocMemory.document_id == doc_id)
                )
            )

            return doc_mem
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e


    async def _get_conversation_messages(self, session_id: uuid.UUID) -> list[ChatHistory]: 
        """Get user messages and assistant messages of a session"""
        try:
            conversation = (
                await self.db.scalars(
                    select(ChatHistory)
                    .where(
                        ChatHistory.type == "message",
                        ChatHistory.session_id == session_id
                    )
                    .order_by(ChatHistory.created_at.asc())
                )
            ).all()

            return conversation
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e

    async def _llm_summarize(self, messages: list[dict[str, str]]) -> str:
        """Distill memory for document, user or session based on system prompt and conversation"""
        try:
            response = await self.client.chat.completions.create(
                model=self.small_model,
                messages=messages,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Summarize failed: {e}")


    async def _build_exchanges(
        self, 
        session_id: uuid.UUID, 
        system_prompt: str,
        format_param: dict = {}
    ) -> list[dict[str, str]]:
        """Build messages according to specific object to distill memory"""

        conversation_messages = await self._get_conversation_messages(session_id)
        exchanges = "\n".join([f"{message.role} : {message.content["text"]}" for message in conversation_messages])
        format_param["exchanges"] = exchanges
        messages = [{
            "role": "user", 
            "content": system_prompt.format(**format_param)
        }]
        return messages
        

    async def _update_doc(self, doc_id: uuid.UUID, session_id: uuid.UUID) -> PerDocMemory: 
        """Update user's current state about a specific document."""
        current_doc_mem = await self._get_doc_memory(doc_id)
        doc_mem_content = current_doc_mem.content if current_doc_mem else "No information"


        format_param = {
            "current_memory": doc_mem_content
        }
        message = await self._build_exchanges(
            session_id=session_id,
            system_prompt=prompts.DOC_UPDATE_PROMPT,
            format_param=format_param
        )
        summarization = await self._llm_summarize(message)

        try:
            if not current_doc_mem:
                # insert
                doc_mem = PerDocMemory(
                    document_id=doc_id,
                    content=summarization,
                )
                self.db.add(doc_mem)

            else:
                # update
                doc_mem = current_doc_mem
                doc_mem.content = summarization        

            await self.db.commit()
            await self.db.refresh(doc_mem)

            return doc_mem
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e


    async def _update_user_profile(self, session_id: uuid.UUID) -> None: 
        """Update intellectual fingerprint of user."""
        user_profile = self.user_profile_path.read_text(encoding='utf-8')
        format_param = {
            "current_profile": user_profile
        }

        message = await self._build_exchanges(
            session_id=session_id,
            system_prompt=prompts.USER_PROFILE_UPDATE_PROMPT,
            format_param=format_param
        )
        summarization = await self._llm_summarize(message)

        self.user_profile_path.write_text(summarization)

    
    async def _should_retrieve(self, message: str) -> tuple[bool, bool]: 
        """
        Decide when we should retrieve information of 
        a session or a document based on current message.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.small_model,
                messages=[{
                    "role": "user",
                    "content": prompts.RETRIEVAL_GATE_PROMPT.format(message=message)
                }],
                max_tokens=600
            )

            text = response.choices[0].message.content
            if "```" in text:
                text = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(text)

            return (
                bool(data['retrieve_user']),
                bool(data['retrieve_doc']),
            )
        except Exception:
            return True, True


    async def _should_consolidate(self, session_id: uuid.UUID) -> bool:
        """Return True if have enough message to consolidate"""
        count = await self.db.scalar(
            select(func.count(ChatHistory.id))
            .where(ChatHistory.session_id == session_id)
            .where(ChatHistory.is_consolidated == False)
            .where(ChatHistory.role.in_(["user", "assistant"]))
        )

        return count >= CONSOLIDATE_EVERY_N


    async def _mark_consolidated(self, session_id: uuid.UUID) -> None:
        """Mark messages as consolidated after summarizing"""
        await self.db.execute(
            update(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .where(ChatHistory.is_consolidated == False)
            .where(ChatHistory.role.in_(["user", "assistant"]))
            .values(is_consolidated=True)
        )
        await self.db.commit()

    
    async def before_run(
        self, 
        message: str, 
        doc_id: uuid.UUID | None
    ) -> dict: 
        """Retrieve pre-run long-term memory context (user profile & per-doc memory).

        Args:
            message (str): Current user question message.
            doc_id (uuid.UUID | None): Active document UUID if scoped.

        Returns:
            dict: Context dictionary containing 'user_profile' and/or 'doc_memory' strings.
        """
        logger.debug(f"Retrieving memory context before run for message snippet: {message[:50]}...")
        retrieve_user, retrieve_doc = await self._should_retrieve(message)

        context = {}
        if retrieve_user:
            context["user_profile"] = self._get_user_profile()
        if retrieve_doc and doc_id:
            context["doc_memory"] = await self._get_doc_memory_content(doc_id)

        logger.debug(f"Memory context retrieved: user_profile={bool(context.get('user_profile'))}, doc_memory={bool(context.get('doc_memory'))}")
        return context


    async def after_run(self, session_id: uuid.UUID, doc_ids_used: list[uuid.UUID]):
        try:
            should_consolidate = await self._should_consolidate(session_id)
            if should_consolidate:
                logger.info(f"[Session {session_id}] Memory consolidation triggered")
                for doc_id in doc_ids_used:
                    await self._update_doc(doc_id=doc_id, session_id=session_id)
                await self._update_user_profile(session_id)
                await self._mark_consolidated(session_id)
                logger.info(f"[Session {session_id}] Memory consolidation completed")
        except Exception as e:
            logger.warning(f"[Session {session_id}] Memory consolidation failed: {e}")
    


