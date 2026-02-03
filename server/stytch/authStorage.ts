/**
 * Minimal Auth Storage for Cloud Run
 * 
 * Only includes user operations needed for authentication.
 * Avoids importing the entire storage module which has heavy dependencies.
 */
import { db } from "../db";
import { users } from "@shared/schema";
import { eq } from "drizzle-orm";

export interface AuthUser {
  id: number;
  email: string;
  firstName: string | null;
  lastName: string | null;
  role: string;
  phone: string | null;
}

export interface CreateUserData {
  email: string;
  firstName?: string;
  lastName?: string;
  role: string;
  phone?: string;
}

export const authStorage = {
  async getUserByEmail(email: string): Promise<AuthUser | null> {
    try {
      const result = await db
        .select({
          id: users.id,
          email: users.email,
          firstName: users.firstName,
          lastName: users.lastName,
          role: users.role,
          phone: users.phone,
        })
        .from(users)
        .where(eq(users.email, email))
        .limit(1);
      
      return result[0] || null;
    } catch (error) {
      console.error("[AUTH_STORAGE] getUserByEmail error:", error);
      return null;
    }
  },

  async getUserByPhone(phone: string): Promise<AuthUser | null> {
    try {
      const result = await db
        .select({
          id: users.id,
          email: users.email,
          firstName: users.firstName,
          lastName: users.lastName,
          role: users.role,
          phone: users.phone,
        })
        .from(users)
        .where(eq(users.phone, phone))
        .limit(1);
      
      return result[0] || null;
    } catch (error) {
      console.error("[AUTH_STORAGE] getUserByPhone error:", error);
      return null;
    }
  },

  async createUser(data: CreateUserData): Promise<AuthUser | null> {
    try {
      const result = await db
        .insert(users)
        .values({
          email: data.email,
          firstName: data.firstName || null,
          lastName: data.lastName || null,
          role: data.role,
          phone: data.phone || null,
        })
        .returning({
          id: users.id,
          email: users.email,
          firstName: users.firstName,
          lastName: users.lastName,
          role: users.role,
          phone: users.phone,
        });
      
      return result[0] || null;
    } catch (error) {
      console.error("[AUTH_STORAGE] createUser error:", error);
      return null;
    }
  },
};
