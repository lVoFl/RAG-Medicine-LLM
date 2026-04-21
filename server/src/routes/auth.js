import { Router } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import pool from "../db.js";

const router = Router();

const JWT_SECRET = process.env.JWT_SECRET || "change-this-secret-in-production";
const SALT_ROUNDS = 10;

// POST /api/auth/register
router.post("/register", async (req, res, next) => {
  try {
    const { username, password, email } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: "username and password are required" });
    }
    if (password.length < 6) {
      return res.status(400).json({ error: "password must be at least 6 characters" });
    }

    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);

    const adminUsernames = (process.env.ADMIN_USERNAMES || "admin")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const isAdmin = adminUsernames.includes(username.trim());

    const result = await pool.query(
      "INSERT INTO users (username, password_hash, email, is_admin) VALUES ($1, $2, $3, $4) RETURNING id, is_admin",
      [username, passwordHash, email ?? null, isAdmin]
    );

    const userId = result.rows[0].id;
    const token = jwt.sign({ id: userId, username, isAdmin: result.rows[0].is_admin }, JWT_SECRET, {
      expiresIn: "7d",
    });
    res.status(201).json({ token, username, isAdmin: result.rows[0].is_admin });
  } catch (err) {
    if (err.code === "23505") {
      return res.status(409).json({ error: "username already exists" });
    }
    next(err);
  }
});

// POST /api/auth/login
router.post("/login", async (req, res, next) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: "username and password are required" });
    }

    const result = await pool.query(
      "SELECT id, username, password_hash, is_admin FROM users WHERE username = $1",
      [username]
    );

    const user = result.rows[0];
    if (!user) {
      return res.status(401).json({ error: "invalid username or password" });
    }

    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) {
      return res.status(401).json({ error: "invalid username or password" });
    }

    const token = jwt.sign({ id: user.id, username: user.username, isAdmin: user.is_admin }, JWT_SECRET, {
      expiresIn: "7d",
    });
    res.json({ token, username: user.username, isAdmin: user.is_admin });
  } catch (err) {
    next(err);
  }
});

export default router;
