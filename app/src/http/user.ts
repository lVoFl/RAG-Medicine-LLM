import request from "./request";
import type { LoginRequest, RegisterRequest } from "../types/user"

export const login = (data: LoginRequest) => {
  return request.post("/api/auth/login", data);
};

export const register = (data: RegisterRequest) => {
    return request.post("/api/auth/register", data);
}