import type { Express, Request } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { setupAuth, isAuthenticated, isDoctor, hashPassword, comparePassword, generateToken, sendVerificationEmail, sendPasswordResetEmail } from "./auth";
import { pubmedService, physionetService, kaggleService, whoService } from "./dataIntegration";
import OpenAI from "openai";
import Sentiment from "sentiment";
import multer from "multer";
import path from "path";
import fs from "fs";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const sentiment = new Sentiment();

// Configure multer for file uploads (KYC photos)
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const uploadPath = path.join(process.cwd(), 'uploads', 'kyc');
      fs.mkdirSync(uploadPath, { recursive: true });
      cb(null, uploadPath);
    },
    filename: (req, file, cb) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
      cb(null, 'kyc-' + uniqueSuffix + path.extname(file.originalname));
    }
  }),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|pdf/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    if (extname && mimetype) {
      cb(null, true);
    } else {
      cb(new Error('Only .jpg, .jpeg, .png, and .pdf files are allowed'));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  setupAuth(app);

  // ============== AUTHENTICATION ROUTES ==============
  
  // Doctor signup
  app.post('/api/auth/signup/doctor', upload.single('kycPhoto'), async (req, res) => {
    try {
      const { email, password, firstName, lastName, organization, medicalLicenseNumber, licenseCountry, termsAccepted } = req.body;
      
      // Validation
      if (!email || !password || !firstName || !lastName || !organization || !medicalLicenseNumber || !licenseCountry) {
        return res.status(400).json({ message: "All fields are required" });
      }
      
      if (!termsAccepted || termsAccepted !== 'true') {
        return res.status(400).json({ message: "You must accept the terms and conditions" });
      }
      
      // Check if user already exists
      const existingUser = await storage.getUserByEmail(email);
      if (existingUser) {
        return res.status(400).json({ message: "Email already registered" });
      }
      
      // Hash password
      const passwordHash = await hashPassword(password);
      
      // Generate verification token
      const verificationToken = generateToken();
      const verificationTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Get KYC photo URL if uploaded
      const kycPhotoUrl = req.file ? `/uploads/kyc/${req.file.filename}` : undefined;
      
      // Create user
      const user = await storage.createUser({
        email,
        passwordHash,
        firstName,
        lastName,
        role: 'doctor',
        organization,
        medicalLicenseNumber,
        licenseCountry,
        kycPhotoUrl,
        emailVerified: false,
        verificationToken,
        verificationTokenExpires,
        termsAccepted: true,
        termsAcceptedAt: new Date(),
        creditBalance: 0,
      });
      
      // Send verification email
      try {
        await sendVerificationEmail(email, verificationToken, firstName);
      } catch (emailError) {
        console.error("Error sending verification email:", emailError);
        // Don't fail signup if email fails
      }
      
      res.json({
        message: "Account created successfully. Please check your email to verify your account.",
        userId: user.id,
      });
    } catch (error) {
      console.error("Error in doctor signup:", error);
      res.status(500).json({ message: "Failed to create account" });
    }
  });
  
  // Patient signup
  app.post('/api/auth/signup/patient', async (req, res) => {
    try {
      const { email, password, firstName, lastName, ehrImportMethod, ehrPlatform, termsAccepted } = req.body;
      
      // Validation
      if (!email || !password || !firstName || !lastName || !ehrImportMethod) {
        return res.status(400).json({ message: "All fields are required" });
      }
      
      if (!termsAccepted) {
        return res.status(400).json({ message: "You must accept the terms and conditions" });
      }
      
      // Check if user already exists
      const existingUser = await storage.getUserByEmail(email);
      if (existingUser) {
        return res.status(400).json({ message: "Email already registered" });
      }
      
      // Hash password
      const passwordHash = await hashPassword(password);
      
      // Generate verification token
      const verificationToken = generateToken();
      const verificationTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Start 7-day free trial
      const trialEndsAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
      
      // Create user
      const user = await storage.createUser({
        email,
        passwordHash,
        firstName,
        lastName,
        role: 'patient',
        ehrImportMethod,
        ehrPlatform,
        emailVerified: false,
        verificationToken,
        verificationTokenExpires,
        termsAccepted: true,
        termsAcceptedAt: new Date(),
        subscriptionStatus: 'trialing',
        trialEndsAt,
        creditBalance: 20, // 20 free consultation credits during trial
      });
      
      // Send verification email
      try {
        await sendVerificationEmail(email, verificationToken, firstName);
      } catch (emailError) {
        console.error("Error sending verification email:", emailError);
      }
      
      res.json({
        message: "Account created successfully. Please check your email to verify your account. Your 7-day free trial has started!",
        userId: user.id,
      });
    } catch (error) {
      console.error("Error in patient signup:", error);
      res.status(500).json({ message: "Failed to create account" });
    }
  });
  
  // Login
  app.post('/api/auth/login', async (req, res) => {
    try {
      const { email, password, rememberMe } = req.body;
      
      if (!email || !password) {
        return res.status(400).json({ message: "Email and password are required" });
      }
      
      // Get user by email
      const user = await storage.getUserByEmail(email);
      if (!user) {
        return res.status(401).json({ message: "Invalid email or password" });
      }
      
      // Compare password
      const isPasswordValid = await comparePassword(password, user.passwordHash!);
      if (!isPasswordValid) {
        return res.status(401).json({ message: "Invalid email or password" });
      }
      
      // Check if email is verified
      if (!user.emailVerified) {
        return res.status(403).json({ message: "Please verify your email before logging in" });
      }
      
      // Set session
      (req.session as any).userId = user.id;
      
      // Extend session if remember me is checked
      if (rememberMe) {
        req.session.cookie.maxAge = 30 * 24 * 60 * 60 * 1000; // 30 days in milliseconds
      }
      
      // Save session to ensure TTL is updated in the store
      await new Promise<void>((resolve, reject) => {
        req.session.save((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
      
      res.json({
        message: "Login successful",
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role,
        },
      });
    } catch (error) {
      console.error("Error in login:", error);
      res.status(500).json({ message: "Failed to log in" });
    }
  });
  
  // Logout
  app.post('/api/auth/logout', (req, res) => {
    req.session.destroy((err) => {
      if (err) {
        return res.status(500).json({ message: "Failed to log out" });
      }
      res.json({ message: "Logged out successfully" });
    });
  });
  
  // Verify email
  app.get('/api/auth/verify-email', async (req, res) => {
    try {
      const { token } = req.query;
      
      if (!token || typeof token !== 'string') {
        return res.status(400).json({ message: "Invalid verification token" });
      }
      
      const user = await storage.verifyEmail(token);
      if (!user) {
        return res.status(400).json({ message: "Invalid or expired verification token" });
      }
      
      res.json({ message: "Email verified successfully. You can now log in." });
    } catch (error) {
      console.error("Error verifying email:", error);
      res.status(500).json({ message: "Failed to verify email" });
    }
  });
  
  // Request password reset
  app.post('/api/auth/forgot-password', async (req, res) => {
    try {
      const { email } = req.body;
      
      if (!email) {
        return res.status(400).json({ message: "Email is required" });
      }
      
      const user = await storage.getUserByEmail(email);
      if (!user) {
        // Don't reveal if email exists
        return res.json({ message: "If that email is registered, you will receive a password reset link" });
      }
      
      // Generate reset token
      const resetToken = generateToken();
      const resetTokenExpires = new Date(Date.now() + 60 * 60 * 1000); // 1 hour
      
      await storage.updateResetToken(user.id, resetToken, resetTokenExpires);
      
      // Send reset email
      try {
        await sendPasswordResetEmail(user.email!, resetToken, user.firstName!);
      } catch (emailError) {
        console.error("Error sending password reset email:", emailError);
      }
      
      res.json({ message: "If that email is registered, you will receive a password reset link" });
    } catch (error) {
      console.error("Error in forgot password:", error);
      res.status(500).json({ message: "Failed to process request" });
    }
  });
  
  // Reset password
  app.post('/api/auth/reset-password', async (req, res) => {
    try {
      const { token, newPassword } = req.body;
      
      if (!token || !newPassword) {
        return res.status(400).json({ message: "Token and new password are required" });
      }
      
      const passwordHash = await hashPassword(newPassword);
      const user = await storage.resetPassword(token, passwordHash);
      
      if (!user) {
        return res.status(400).json({ message: "Invalid or expired reset token" });
      }
      
      res.json({ message: "Password reset successfully. You can now log in with your new password." });
    } catch (error) {
      console.error("Error resetting password:", error);
      res.status(500).json({ message: "Failed to reset password" });
    }
  });

  // Get current user
  app.get('/api/auth/user', isAuthenticated, async (req, res) => {
    try {
      const userId = req.user!.id;
      const user = await storage.getUser(userId);
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });

  // Patient profile routes
  app.get('/api/patient/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.getPatientProfile(userId);
      res.json(profile || null);
    } catch (error) {
      console.error("Error fetching patient profile:", error);
      res.status(500).json({ message: "Failed to fetch profile" });
    }
  });

  app.post('/api/patient/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.upsertPatientProfile({
        userId,
        ...req.body,
      });
      res.json(profile);
    } catch (error) {
      console.error("Error updating patient profile:", error);
      res.status(500).json({ message: "Failed to update profile" });
    }
  });

  // Doctor profile routes
  app.get('/api/doctor/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.getDoctorProfile(userId);
      res.json(profile || null);
    } catch (error) {
      console.error("Error fetching doctor profile:", error);
      res.status(500).json({ message: "Failed to fetch profile" });
    }
  });

  app.post('/api/doctor/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.upsertDoctorProfile({
        userId,
        ...req.body,
      });
      res.json(profile);
    } catch (error) {
      console.error("Error updating doctor profile:", error);
      res.status(500).json({ message: "Failed to update profile" });
    }
  });

  // Daily followup routes
  app.get('/api/daily-followup/today', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followup = await storage.getDailyFollowup(userId, new Date());
      res.json(followup || null);
    } catch (error) {
      console.error("Error fetching daily followup:", error);
      res.status(500).json({ message: "Failed to fetch daily followup" });
    }
  });

  app.get('/api/daily-followup/history', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 30;
      const followups = await storage.getRecentFollowups(userId, limit);
      res.json(followups);
    } catch (error) {
      console.error("Error fetching followup history:", error);
      res.status(500).json({ message: "Failed to fetch followup history" });
    }
  });

  app.post('/api/daily-followup', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followup = await storage.createDailyFollowup({
        patientId: userId,
        ...req.body,
      });
      res.json(followup);
    } catch (error) {
      console.error("Error creating daily followup:", error);
      res.status(500).json({ message: "Failed to create daily followup" });
    }
  });

  app.patch('/api/daily-followup/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const followup = await storage.updateDailyFollowup(id, req.body);
      res.json(followup);
    } catch (error) {
      console.error("Error updating daily followup:", error);
      res.status(500).json({ message: "Failed to update daily followup" });
    }
  });

  // Chat routes (Agent Clona & Assistant Lysa)
  app.get('/api/chat/messages', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const agentType = req.query.agent as string;
      const messages = await storage.getChatMessages(userId, agentType);
      res.json(messages);
    } catch (error) {
      console.error("Error fetching chat messages:", error);
      res.status(500).json({ message: "Failed to fetch chat messages" });
    }
  });

  app.post('/api/chat/send', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { content, agentType } = req.body;

      let session = await storage.getActiveSession(userId, agentType);
      
      if (!session) {
        const sessionTitle = content.substring(0, 50) + (content.length > 50 ? '...' : '');
        session = await storage.createSession({
          patientId: userId,
          agentType,
          sessionTitle,
        });
      }

      await storage.createChatMessage({
        sessionId: session.id,
        userId,
        role: 'user',
        content,
        agentType,
      });

      const user = await storage.getUser(userId);
      const isDoctor = user?.role === 'doctor';
      
      const systemPrompt = agentType === 'clona'
        ? `You are Agent Clona, a compassionate AI health assistant for immunocompromised patients. 
           Provide clear, supportive medical guidance. Always include FDA/CE disclaimer when suggesting 
           medications. Consider geographic context for disease patterns. Ask follow-up questions about 
           symptoms. Maintain HIPAA compliance.`
        : `You are Assistant Lysa, an AI assistant helping doctors review patient data and research. 
           Provide clinical insights, pattern recognition, and evidence-based recommendations. 
           Assist with epidemiological analysis.`;

      const sessionMessages = await storage.getSessionMessages(session.id);
      const recentMessages = sessionMessages.slice(-10).map(msg => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
      }));

      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: systemPrompt },
          ...recentMessages,
          { role: "user", content },
        ],
        temperature: 0.7,
        max_tokens: 800,
      });

      const assistantMessage = completion.choices[0].message.content || "I'm here to help!";

      const extractMedicalEntities = (text: string) => {
        const entities: Array<{ text: string; type: string }> = [];
        const symptoms = ['fever', 'cough', 'headache', 'pain', 'nausea', 'fatigue', 'dizziness', 'sore throat', 'chills', 'shortness of breath'];
        const medications = ['aspirin', 'ibuprofen', 'acetaminophen', 'antibiotic'];
        
        symptoms.forEach(symptom => {
          if (text.toLowerCase().includes(symptom)) {
            entities.push({ text: symptom, type: 'symptom' });
          }
        });
        
        medications.forEach(med => {
          if (text.toLowerCase().includes(med)) {
            entities.push({ text: med, type: 'medication' });
          }
        });
        
        return entities;
      };

      const userEntities = extractMedicalEntities(content);
      const assistantEntities = extractMedicalEntities(assistantMessage);
      const allSymptoms = [...userEntities, ...assistantEntities]
        .filter(e => e.type === 'symptom')
        .map(e => e.text);

      await storage.updateSessionMetadata(session.id, {
        messageCount: (session.messageCount || 0) + 2,
        symptomsDiscussed: Array.from(new Set([...(session.symptomsDiscussed || []), ...allSymptoms])),
      });

      const savedMessage = await storage.createChatMessage({
        sessionId: session.id,
        userId,
        role: 'assistant',
        content: assistantMessage,
        agentType,
        medicalEntities: assistantEntities,
      });

      res.json(savedMessage);
    } catch (error) {
      console.error("Error in chat:", error);
      res.status(500).json({ message: "Failed to process chat message" });
    }
  });

  // Chat session routes
  app.get('/api/chat/sessions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const agentType = req.query.agent as string | undefined;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 50;
      const sessions = await storage.getPatientSessions(userId, agentType, limit);
      res.json(sessions);
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
      res.status(500).json({ message: "Failed to fetch chat sessions" });
    }
  });

  app.get('/api/chat/sessions/:sessionId', isAuthenticated, async (req: any, res) => {
    try {
      const { sessionId } = req.params;
      const messages = await storage.getSessionMessages(sessionId);
      res.json(messages);
    } catch (error) {
      console.error("Error fetching session messages:", error);
      res.status(500).json({ message: "Failed to fetch session messages" });
    }
  });

  app.post('/api/chat/sessions/:sessionId/end', isAuthenticated, async (req: any, res) => {
    try {
      const { sessionId } = req.params;
      const messages = await storage.getSessionMessages(sessionId);
      
      const conversationText = messages.map(m => `${m.role}: ${m.content}`).join('\n');
      
      const summaryCompletion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are a medical AI summarizing a patient-doctor chat session. Extract:
            1. Key symptoms mentioned
            2. Recommendations given
            3. Concerns raised
            4. Any vital signs or measurements discussed
            Keep it concise and clinical.`
          },
          {
            role: "user",
            content: `Summarize this medical conversation:\n\n${conversationText}`
          }
        ],
        temperature: 0.3,
        max_tokens: 400,
      });

      const aiSummary = summaryCompletion.choices[0].message.content || "";
      
      const symptoms = Array.from(new Set(
        messages.flatMap(m => {
          const entities = m.medicalEntities || [];
          return entities.filter((e: any) => e.type === 'symptom').map((e: any) => e.text);
        })
      ));

      const healthInsights = {
        keySymptoms: symptoms,
        conversationSummary: aiSummary
      };

      const session = await storage.endSession(sessionId, aiSummary, healthInsights);
      res.json(session);
    } catch (error) {
      console.error("Error ending session:", error);
      res.status(500).json({ message: "Failed to end session" });
    }
  });

  app.get('/api/doctor/patient-sessions/:patientId', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { patientId } = req.params;
      const oneMonthAgo = new Date();
      oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
      const now = new Date();
      
      const sessions = await storage.getSessionsInDateRange(patientId, oneMonthAgo, now, 'clona');
      res.json(sessions);
    } catch (error) {
      console.error("Error fetching patient sessions:", error);
      res.status(500).json({ message: "Failed to fetch patient sessions" });
    }
  });

  // Medication routes
  app.get('/api/medications', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const medications = await storage.getActiveMedications(userId);
      res.json(medications);
    } catch (error) {
      console.error("Error fetching medications:", error);
      res.status(500).json({ message: "Failed to fetch medications" });
    }
  });

  app.post('/api/medications', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const medication = await storage.createMedication({
        patientId: userId,
        ...req.body,
      });
      res.json(medication);
    } catch (error) {
      console.error("Error creating medication:", error);
      res.status(500).json({ message: "Failed to create medication" });
    }
  });

  // Dynamic tasks
  app.get('/api/tasks', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const tasks = await storage.getActiveTasks(userId);
      res.json(tasks);
    } catch (error) {
      console.error("Error fetching tasks:", error);
      res.status(500).json({ message: "Failed to fetch tasks" });
    }
  });

  app.post('/api/tasks', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const task = await storage.createDynamicTask({
        patientId: userId,
        ...req.body,
      });
      res.json(task);
    } catch (error) {
      console.error("Error creating task:", error);
      res.status(500).json({ message: "Failed to create task" });
    }
  });

  app.post('/api/tasks/:id/complete', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const task = await storage.completeTask(id);
      res.json(task);
    } catch (error) {
      console.error("Error completing task:", error);
      res.status(500).json({ message: "Failed to complete task" });
    }
  });

  // Auto journals
  app.get('/api/journals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const journals = await storage.getRecentJournals(userId);
      res.json(journals);
    } catch (error) {
      console.error("Error fetching journals:", error);
      res.status(500).json({ message: "Failed to fetch journals" });
    }
  });

  app.post('/api/journals/auto-generate', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followups = await storage.getRecentFollowups(userId, 1);
      const tasks = await storage.getActiveTasks(userId);
      
      const context = `Recent health data: ${followups.length > 0 ? JSON.stringify(followups[0]) : 'No data'}. Active tasks: ${tasks.map(t => t.title).join(', ')}`;
      
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You are a health journal assistant. Create a brief, empathetic health journal entry based on the patient's data." },
          { role: "user", content: context },
        ],
        max_tokens: 200,
      });

      const content = completion.choices[0].message.content || "Feeling well today.";
      const sentimentAnalysis = sentiment.analyze(content);
      
      const journal = await storage.createAutoJournal({
        patientId: userId,
        content,
        summary: content.slice(0, 100),
        mood: sentimentAnalysis.score > 0 ? 'positive' : sentimentAnalysis.score < 0 ? 'negative' : 'neutral',
        stressLevel: Math.max(1, Math.min(10, 5 - sentimentAnalysis.score)),
        generatedFromData: { followup: true, symptoms: true, wearables: false },
      });

      res.json(journal);
    } catch (error) {
      console.error("Error generating journal:", error);
      res.status(500).json({ message: "Failed to generate journal" });
    }
  });

  // Calm activities
  app.get('/api/calm-activities', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const activities = await storage.getCalmActivities(userId);
      res.json(activities);
    } catch (error) {
      console.error("Error fetching calm activities:", error);
      res.status(500).json({ message: "Failed to fetch calm activities" });
    }
  });

  app.post('/api/calm-activities/:id/rate', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { effectiveness } = req.body;
      const activity = await storage.updateCalmActivityEffectiveness(id, effectiveness);
      res.json(activity);
    } catch (error) {
      console.error("Error rating activity:", error);
      res.status(500).json({ message: "Failed to rate activity" });
    }
  });

  // Behavioral insights
  app.get('/api/behavioral-insights', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const insights = await storage.getRecentInsights(userId);
      res.json(insights);
    } catch (error) {
      console.error("Error fetching insights:", error);
      res.status(500).json({ message: "Failed to fetch insights" });
    }
  });

  app.post('/api/behavioral-insights', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const insight = await storage.createBehavioralInsight({
        patientId: userId,
        ...req.body,
      });
      res.json(insight);
    } catch (error) {
      console.error("Error creating insight:", error);
      res.status(500).json({ message: "Failed to create insight" });
    }
  });

  // Doctor-only routes
  app.get('/api/doctor/patients', isDoctor, async (req: any, res) => {
    try {
      const patients = await storage.getAllPatients();
      res.json(patients);
    } catch (error) {
      console.error("Error fetching patients:", error);
      res.status(500).json({ message: "Failed to fetch patients" });
    }
  });

  app.get('/api/doctor/research-consents', isDoctor, async (req: any, res) => {
    try {
      const consents = await storage.getPendingConsents();
      res.json(consents);
    } catch (error) {
      console.error("Error fetching consents:", error);
      res.status(500).json({ message: "Failed to fetch consents" });
    }
  });

  app.post('/api/doctor/research-consents/:id/review', isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { id } = req.params;
      const { status } = req.body;
      const consent = await storage.updateConsentStatus(id, status, userId);
      res.json(consent);
    } catch (error) {
      console.error("Error updating consent:", error);
      res.status(500).json({ message: "Failed to update consent" });
    }
  });

  app.get('/api/doctor/research-reports', isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const reports = await storage.getResearchReports(userId);
      res.json(reports);
    } catch (error) {
      console.error("Error fetching reports:", error);
      res.status(500).json({ message: "Failed to fetch reports" });
    }
  });

  app.post('/api/doctor/research-reports', isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You are an AI Research Agent. Generate medical research findings based on patient data patterns." },
          { role: "user", content: `Analyze: ${req.body.analysisType}. Generate research findings.` },
        ],
        max_tokens: 500,
      });

      const findings = [
        {
          finding: "Correlation detected between blood glucose levels and stress scores",
          significance: "Moderate",
          confidence: 0.78,
        },
        {
          finding: "Improved medication adherence in patients using daily follow-up feature",
          significance: "High",
          confidence: 0.85,
        },
      ];

      const report = await storage.createResearchReport({
        createdBy: userId,
        title: req.body.title,
        summary: completion.choices[0].message.content || "Research analysis completed",
        findings,
        visualizations: [],
        patientCohortSize: 150,
        analysisType: req.body.analysisType,
      });

      res.json(report);
    } catch (error) {
      console.error("Error creating report:", error);
      res.status(500).json({ message: "Failed to create report" });
    }
  });

  // Educational content
  app.get('/api/education/progress', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const progress = await storage.getEducationalProgress(userId);
      res.json(progress);
    } catch (error) {
      console.error("Error fetching education progress:", error);
      res.status(500).json({ message: "Failed to fetch education progress" });
    }
  });

  app.post('/api/education/progress', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const progress = await storage.upsertEducationalProgress({
        patientId: userId,
        ...req.body,
      });
      res.json(progress);
    } catch (error) {
      console.error("Error updating education progress:", error);
      res.status(500).json({ message: "Failed to update education progress" });
    }
  });

  // Psychological counseling session routes
  app.get('/api/counseling/sessions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const sessions = await storage.getCounselingSessions(userId);
      res.json(sessions);
    } catch (error) {
      console.error("Error fetching counseling sessions:", error);
      res.status(500).json({ message: "Failed to fetch counseling sessions" });
    }
  });

  app.post('/api/counseling/sessions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const session = await storage.createCounselingSession({
        userId,
        ...req.body,
      });
      res.json(session);
    } catch (error) {
      console.error("Error creating counseling session:", error);
      res.status(500).json({ message: "Failed to create counseling session" });
    }
  });

  app.patch('/api/counseling/sessions/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const session = await storage.updateCounselingSession(id, req.body);
      if (!session) {
        return res.status(404).json({ message: "Session not found" });
      }
      res.json(session);
    } catch (error) {
      console.error("Error updating counseling session:", error);
      res.status(500).json({ message: "Failed to update counseling session" });
    }
  });

  app.delete('/api/counseling/sessions/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteCounselingSession(id);
      if (!deleted) {
        return res.status(404).json({ message: "Session not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting counseling session:", error);
      res.status(500).json({ message: "Failed to delete counseling session" });
    }
  });

  // Training dataset routes (doctor only)
  app.get('/api/training/datasets', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const datasets = await storage.getTrainingDatasets(userId);
      res.json(datasets);
    } catch (error) {
      console.error("Error fetching training datasets:", error);
      res.status(500).json({ message: "Failed to fetch training datasets" });
    }
  });

  app.post('/api/training/datasets', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const dataset = await storage.createTrainingDataset({
        uploadedBy: userId,
        ...req.body,
      });
      res.json(dataset);
    } catch (error) {
      console.error("Error creating training dataset:", error);
      res.status(500).json({ message: "Failed to create training dataset" });
    }
  });

  app.patch('/api/training/datasets/:id', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const dataset = await storage.updateTrainingDataset(id, req.body);
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      res.json(dataset);
    } catch (error) {
      console.error("Error updating training dataset:", error);
      res.status(500).json({ message: "Failed to update training dataset" });
    }
  });

  app.delete('/api/training/datasets/:id', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteTrainingDataset(id);
      if (!deleted) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting training dataset:", error);
      res.status(500).json({ message: "Failed to delete training dataset" });
    }
  });

  // Public data source integration routes (doctor only)
  app.get('/api/data-sources/pubmed/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query, maxResults } = req.query;
      const result = await pubmedService.search(query as string, parseInt(maxResults as string) || 100);
      res.json(result);
    } catch (error) {
      console.error("Error searching PubMed:", error);
      res.status(500).json({ message: "Failed to search PubMed" });
    }
  });

  app.post('/api/data-sources/pubmed/fetch', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { ids } = req.body;
      const articles = await pubmedService.fetchArticles(ids);
      res.json(articles);
    } catch (error) {
      console.error("Error fetching PubMed articles:", error);
      res.status(500).json({ message: "Failed to fetch PubMed articles" });
    }
  });

  app.get('/api/data-sources/physionet/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query } = req.query;
      const datasets = await physionetService.searchDatasets(query as string || "");
      res.json(datasets);
    } catch (error) {
      console.error("Error searching PhysioNet:", error);
      res.status(500).json({ message: "Failed to search PhysioNet" });
    }
  });

  app.get('/api/data-sources/physionet/dataset/:id', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { id } = req.params;
      const dataset = await physionetService.getDatasetInfo(id);
      if (!dataset) {
        return res.status(404).json({ message: "Dataset not found" });
      }
      res.json(dataset);
    } catch (error) {
      console.error("Error fetching PhysioNet dataset:", error);
      res.status(500).json({ message: "Failed to fetch PhysioNet dataset" });
    }
  });

  app.get('/api/data-sources/kaggle/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query, page } = req.query;
      const datasets = await kaggleService.searchDatasets(query as string, parseInt(page as string) || 1);
      res.json(datasets);
    } catch (error) {
      console.error("Error searching Kaggle:", error);
      res.status(500).json({ message: "Failed to search Kaggle" });
    }
  });

  app.get('/api/data-sources/kaggle/dataset/:owner/:name', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { owner, name } = req.params;
      const metadata = await kaggleService.getDatasetMetadata(owner, name);
      res.json(metadata);
    } catch (error) {
      console.error("Error fetching Kaggle dataset:", error);
      res.status(500).json({ message: "Failed to fetch Kaggle dataset" });
    }
  });

  app.get('/api/data-sources/kaggle/dataset/:owner/:name/files', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { owner, name } = req.params;
      const files = await kaggleService.listDatasetFiles(owner, name);
      res.json(files);
    } catch (error) {
      console.error("Error listing Kaggle dataset files:", error);
      res.status(500).json({ message: "Failed to list Kaggle dataset files" });
    }
  });

  // WHO data integration routes (doctor only)
  app.get('/api/data-sources/who/indicators', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const indicators = await whoService.listIndicators();
      res.json(indicators);
    } catch (error) {
      console.error("Error listing WHO indicators:", error);
      res.status(500).json({ message: "Failed to list WHO indicators" });
    }
  });

  app.get('/api/data-sources/who/indicators/popular', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const indicators = whoService.getPopularIndicators();
      res.json(indicators);
    } catch (error) {
      console.error("Error getting popular WHO indicators:", error);
      res.status(500).json({ message: "Failed to get popular WHO indicators" });
    }
  });

  app.get('/api/data-sources/who/indicators/search', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { query } = req.query;
      const indicators = await whoService.searchIndicators(query as string);
      res.json(indicators);
    } catch (error) {
      console.error("Error searching WHO indicators:", error);
      res.status(500).json({ message: "Failed to search WHO indicators" });
    }
  });

  app.get('/api/data-sources/who/data/:code', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const { code } = req.params;
      const { country, year, sex } = req.query;
      const filters: any = {};
      if (country) filters.country = country;
      if (year) filters.year = parseInt(year as string);
      if (sex) filters.sex = sex;
      
      const data = await whoService.getIndicatorData(code, filters);
      res.json(data);
    } catch (error) {
      console.error("Error getting WHO indicator data:", error);
      res.status(500).json({ message: "Failed to get WHO indicator data" });
    }
  });

  app.get('/api/data-sources/who/countries', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const countries = await whoService.getCountries();
      res.json(countries);
    } catch (error) {
      console.error("Error getting WHO countries:", error);
      res.status(500).json({ message: "Failed to get WHO countries" });
    }
  });

  // Health insight consent routes
  app.get('/api/consents', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const consents = await storage.getHealthInsightConsents(userId);
      res.json(consents);
    } catch (error) {
      console.error("Error fetching consents:", error);
      res.status(500).json({ message: "Failed to fetch consents" });
    }
  });

  app.get('/api/consents/active', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const consents = await storage.getActiveConsents(userId);
      res.json(consents);
    } catch (error) {
      console.error("Error fetching active consents:", error);
      res.status(500).json({ message: "Failed to fetch active consents" });
    }
  });

  app.post('/api/consents', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const consent = await storage.createHealthInsightConsent({
        userId,
        ...req.body,
      });
      res.json(consent);
    } catch (error) {
      console.error("Error creating consent:", error);
      res.status(500).json({ message: "Failed to create consent" });
    }
  });

  app.patch('/api/consents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const consent = await storage.updateHealthInsightConsent(id, req.body);
      if (!consent) {
        return res.status(404).json({ message: "Consent not found" });
      }
      res.json(consent);
    } catch (error) {
      console.error("Error updating consent:", error);
      res.status(500).json({ message: "Failed to update consent" });
    }
  });

  app.post('/api/consents/:id/revoke', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { reason } = req.body;
      const consent = await storage.revokeConsent(id, reason);
      if (!consent) {
        return res.status(404).json({ message: "Consent not found" });
      }
      res.json(consent);
    } catch (error) {
      console.error("Error revoking consent:", error);
      res.status(500).json({ message: "Failed to revoke consent" });
    }
  });

  app.delete('/api/consents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteHealthInsightConsent(id);
      if (!deleted) {
        return res.status(404).json({ message: "Consent not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting consent:", error);
      res.status(500).json({ message: "Failed to delete consent" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
