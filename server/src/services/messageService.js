import pool from "../db.js";

function createHttpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

export async function ensureConversationAccessible(conversationId, userId) {
  const result = await pool.query(
    `
    SELECT id
    FROM conversations
    WHERE id = $1 AND user_id = $2 AND is_deleted = FALSE
    `,
    [conversationId, userId]
  );

  if (!result.rows[0]) {
    throw createHttpError(404, "conversation not found");
  }
}

export async function listMessagesByConversation(conversationId) {
  const result = await pool.query(
    `
    SELECT id, conversation_id, role, content, tokens, created_at
    FROM messages
    WHERE conversation_id = $1
    ORDER BY created_at ASC, id ASC
    `,
    [conversationId]
  );
  return result.rows;
}

export async function createMessage({ conversationId, role, content, tokens }) {
  const result = await pool.query(
    `
    INSERT INTO messages (conversation_id, role, content, tokens)
    VALUES ($1, $2, $3, $4)
    RETURNING id, conversation_id, role, content, tokens, created_at
    `,
    [conversationId, role, content, tokens ?? null]
  );
  console.log(result.rows[0]);
  return result.rows[0];
}

export async function patchMessage({ conversationId, messageId, content, tokens }) {
  const nextContent = content === undefined ? null : content;
  const nextTokens = tokens === undefined ? null : tokens;

  const result = await pool.query(
    `
    UPDATE messages
    SET
      content = COALESCE($1::jsonb, content),
      tokens = COALESCE($2::int, tokens)
    WHERE id = $3 AND conversation_id = $4
    RETURNING id, conversation_id, role, content, tokens, created_at
    `,
    [nextContent, nextTokens, messageId, conversationId]
  );

  if (!result.rows[0]) {
    throw createHttpError(404, "message not found");
  }

  return result.rows[0];
}
