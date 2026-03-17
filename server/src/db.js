import pg from "pg";
import dotenv from "dotenv";
dotenv.config();

const poolConfig = process.env.DATABASE_URL
  ? {
      connectionString: process.env.DATABASE_URL,
    }
  : {
      host: process.env.DB_HOST || "localhost",
      port: Number(process.env.DB_PORT) || 5432,
      user: process.env.DB_USER || "postgres",
      password: process.env.DB_PASSWORD || "",
      database: process.env.DB_NAME || "app",
    };

const pool = new pg.Pool({
  ...poolConfig,
  max: 10,
});

// Create users table if it doesn't exist
try {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      username VARCHAR(50) NOT NULL UNIQUE,
      email VARCHAR(255),
      password_hash VARCHAR(255) NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW()
    )
  `);
  await pool.query(`
    ALTER TABLE users
    ADD COLUMN IF NOT EXISTS email VARCHAR(255)
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS conversations (
      id BIGSERIAL PRIMARY KEY,
      user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      title VARCHAR(255),
      last_message JSONB,
      message_count INT DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      is_deleted BOOLEAN DEFAULT FALSE
    )
  `);
  await pool.query(`
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'conversations'
          AND column_name = 'last_message'
          AND data_type = 'text'
      ) THEN
        ALTER TABLE conversations ADD COLUMN IF NOT EXISTS last_message_json JSONB;
        UPDATE conversations
        SET last_message_json = COALESCE(last_message_json, jsonb_build_object('text', last_message));
        ALTER TABLE conversations DROP COLUMN last_message;
        ALTER TABLE conversations RENAME COLUMN last_message_json TO last_message;
      END IF;
    END $$;
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS messages (
      id BIGSERIAL PRIMARY KEY,
      conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role VARCHAR(20) NOT NULL,
      content JSONB NOT NULL,
      tokens INT,
      created_at TIMESTAMPTZ DEFAULT NOW()
    )
  `);
  await pool.query(`
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'messages'
          AND column_name = 'content'
          AND data_type = 'text'
      ) THEN
        ALTER TABLE messages ADD COLUMN IF NOT EXISTS content_json JSONB;
        UPDATE messages
        SET content_json = COALESCE(content_json, jsonb_build_object('text', content));
        ALTER TABLE messages ALTER COLUMN content_json SET NOT NULL;
        ALTER TABLE messages DROP COLUMN content;
        ALTER TABLE messages RENAME COLUMN content_json TO content;
      END IF;
    END $$;
  `);
  await pool.query(`
    CREATE OR REPLACE FUNCTION set_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
      NEW.updated_at = NOW();
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql
  `);
  await pool.query(`
    DROP TRIGGER IF EXISTS trg_conversations_updated_at ON conversations
  `);
  await pool.query(`
    CREATE TRIGGER trg_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at()
  `);
} catch (err) {
  if (err.code === "28P01") {
    throw new Error(
      "PostgreSQL authentication failed (28P01). Check DB_USER/DB_PASSWORD or DATABASE_URL in .env."
    );
  }
  throw err;
}

export default pool;
