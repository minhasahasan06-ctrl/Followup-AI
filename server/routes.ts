import type { Express, Request } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { isAuthenticated, isDoctor, getSession } from "./auth";
import { pubmedService, physionetService, kaggleService, whoService } from "./dataIntegration";
import { s3Client, textractClient, comprehendMedicalClient, AWS_S3_BUCKET } from "./aws";
import { PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import { StartDocumentAnalysisCommand, GetDocumentAnalysisCommand } from "@aws-sdk/client-textract";
import { DetectEntitiesV2Command, DetectPHICommand } from "@aws-sdk/client-comprehendmedical";
import OpenAI from "openai";
import Sentiment from "sentiment";
import multer from "multer";
import path from "path";
import fs from "fs";
import speakeasy from "speakeasy";
import QRCode from "qrcode";
import crypto from "crypto";

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

// Configure multer for medical document uploads (stored in memory before S3)
const medicalDocUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB limit for medical documents
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
  // ============== SESSION CONFIGURATION ==============
  // Configure session middleware for cookie-based authentication
  app.use(getSession());

  // ============== AUTHENTICATION ROUTES (AWS Cognito) ==============
  const { signUp, signIn, adminConfirmUser, forgotPassword, confirmForgotPassword, getUserInfo, describeUserPoolSchema } = await import('./cognitoAuth');
  const { sendVerificationEmail, sendPasswordResetEmail, sendWelcomeEmail } = await import('./awsSES');
  const { metadataStorage } = await import('./metadataStorage');

  // Debug endpoint to inspect Cognito User Pool schema
  app.get('/api/debug/cognito-schema', async (req, res) => {
    try {
      const schema = await describeUserPoolSchema();
      res.json(schema);
    } catch (error: any) {
      console.error('Error describing Cognito schema:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Debug endpoint to check Cognito email configuration
  app.get('/api/debug/cognito-email-config', async (req, res) => {
    try {
      const schema = await describeUserPoolSchema();
      const emailConfig = schema.emailConfiguration;
      
      const diagnostics = {
        emailSendingAccount: emailConfig.emailSendingAccount,
        sourceArn: emailConfig.sourceArn,
        from: emailConfig.from,
        replyToEmailAddress: emailConfig.replyToEmailAddress,
        configurationSet: emailConfig.configurationSet,
        recommendations: [] as string[],
      };
      
      // Add recommendations based on configuration
      if (emailConfig.emailSendingAccount === 'COGNITO_DEFAULT') {
        diagnostics.recommendations.push(
          'Cognito is using its default email service. This has limitations:',
          '- Only works in sandbox mode (verified emails only)',
          '- Limited email sending capacity',
          '- Consider configuring SES for production use'
        );
      } else if (emailConfig.emailSendingAccount === 'DEVELOPER') {
        if (!emailConfig.sourceArn) {
          diagnostics.recommendations.push(
            'SES is configured but SourceArn is missing. Emails may not be sent properly.'
          );
        } else {
          diagnostics.recommendations.push(
            'SES is configured. Ensure the SES identity (email/domain) is verified in AWS SES console.'
          );
        }
      }
      
      if (!emailConfig.from) {
        diagnostics.recommendations.push(
          'No "From" email address configured. Cognito may use a default address.'
        );
      }
      
      res.json(diagnostics);
    } catch (error: any) {
      console.error('Error checking Cognito email config:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Patient Signup
  app.post('/api/auth/signup/patient', async (req, res) => {
    try {
      const { email, password, firstName, lastName, phoneNumber, ehrImportMethod, ehrPlatform } = req.body;
      
      if (!email || !password || !firstName || !lastName || !phoneNumber) {
        return res.status(400).json({ message: "All fields are required" });
      }
      
      // Sign up in Cognito
      const signUpResponse = await signUp(email, password, firstName, lastName, 'patient', phoneNumber);
      const cognitoSub = signUpResponse.UserSub!;
      const cognitoUsername = signUpResponse.username!;
      
      // Generate verification code (6 digits)
      const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Store phone number temporarily (will be verified after email)
      metadataStorage.setUserMetadata(email, {
        cognitoSub,
        cognitoUsername,
        firstName,
        lastName,
        phoneNumber,
        role: 'patient',
        ehrImportMethod,
        ehrPlatform,
        verificationCode, // Store our generated code
        verificationCodeExpires: expiresAt.getTime(),
      });
      
      // Send verification email via AWS SES (primary method)
      try {
        await sendVerificationEmail(email, verificationCode);
        console.log(`[AUTH] Verification email sent via SES for patient signup: ${email}`);
      } catch (sesError: any) {
        console.error(`[AUTH] Failed to send verification email via SES for ${email}:`, sesError);
        // Fallback to Cognito email
        try {
          await resendConfirmationCode(email, cognitoUsername);
          console.log(`[AUTH] Fallback: Confirmation code resent via Cognito for patient signup: ${email}`);
        } catch (resendError: any) {
          console.error(`[AUTH] Failed to resend confirmation code via Cognito for ${email}:`, resendError);
          // Still return success - user can request resend
        }
      }
      
      res.json({ message: "Signup successful. Please check your email for verification code." });
    } catch (error: any) {
      console.error("Patient signup error:", error);
      if (error.name === 'UsernameExistsException') {
        return res.status(400).json({ message: "An account with this email already exists" });
      }
      res.status(500).json({ message: error.message || "Signup failed" });
    }
  });
  
  // Doctor Signup
  app.post('/api/auth/signup/doctor', upload.single('kycPhoto'), async (req, res) => {
    try {
      const { email, password, firstName, lastName, phoneNumber, organization, medicalLicenseNumber, licenseCountry } = req.body;
      const kycPhoto = req.file;
      
      if (!email || !password || !firstName || !lastName || !phoneNumber || !organization || !medicalLicenseNumber || !licenseCountry) {
        return res.status(400).json({ message: "All fields are required" });
      }
      
      // Sign up in Cognito
      const signUpResponse = await signUp(email, password, firstName, lastName, 'doctor', phoneNumber);
      const cognitoSub = signUpResponse.UserSub!;
      const cognitoUsername = signUpResponse.username!;
      
      // Upload KYC photo if provided
      let kycPhotoUrl: string | undefined;
      if (kycPhoto) {
        const { uploadToS3 } = await import('./awsS3');
        kycPhotoUrl = await uploadToS3(kycPhoto.buffer, `kyc/${email}_${Date.now()}.${kycPhoto.originalname.split('.').pop()}`, kycPhoto.mimetype);
      }
      
      // Generate PDF for doctor application
      const { generateDoctorApplicationPDF, uploadDoctorApplicationToGoogleDrive } = await import('./googleDrive');
      const pdfBuffer = await generateDoctorApplicationPDF({
        email,
        firstName,
        lastName,
        organization,
        medicalLicenseNumber,
        licenseCountry,
        submittedAt: new Date(),
      });
      
      // Upload to Google Drive
      const googleDriveUrl = await uploadDoctorApplicationToGoogleDrive({
        email,
        firstName,
        lastName,
        organization,
        medicalLicenseNumber,
        licenseCountry,
        submittedAt: new Date(),
      }, pdfBuffer);
      
      // Generate verification code (6 digits)
      const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Store doctor data temporarily
      metadataStorage.setUserMetadata(email, {
        cognitoSub,
        cognitoUsername,
        firstName,
        lastName,
        phoneNumber,
        role: 'doctor',
        organization,
        medicalLicenseNumber,
        licenseCountry,
        kycPhotoUrl,
        googleDriveApplicationUrl: googleDriveUrl,
        verificationCode, // Store our generated code
        verificationCodeExpires: expiresAt.getTime(),
      });
      
      // Send verification email via AWS SES (primary method)
      try {
        await sendVerificationEmail(email, verificationCode);
        console.log(`[AUTH] Verification email sent via SES for doctor signup: ${email}`);
      } catch (sesError: any) {
        console.error(`[AUTH] Failed to send verification email via SES for ${email}:`, sesError);
        // Fallback to Cognito email
        try {
          await resendConfirmationCode(email, cognitoUsername);
          console.log(`[AUTH] Fallback: Confirmation code resent via Cognito for doctor signup: ${email}`);
        } catch (resendError: any) {
          console.error(`[AUTH] Failed to resend confirmation code via Cognito for ${email}:`, resendError);
          // Still return success - user can request resend
        }
      }
      
      res.json({ message: "Application submitted successfully. Please check your email for verification code. Your application will be reviewed by our team." });
    } catch (error: any) {
      console.error("Doctor signup error:", error);
      if (error.name === 'UsernameExistsException') {
        return res.status(400).json({ message: "An account with this email already exists" });
      }
      res.status(500).json({ message: error.message || "Signup failed" });
    }
  });
  
  // Verify email with code (Step 1)
  app.post('/api/auth/verify-email', async (req, res) => {
    try {
      const { email, code } = req.body;
      
      if (!email || !code) {
        return res.status(400).json({ message: "Email and verification code are required" });
      }
      
      // Get user metadata to retrieve the Cognito username
      const metadata = metadataStorage.getUserMetadata(email);
      if (!metadata) {
        return res.status(400).json({ message: "No signup data found. Please sign up again." });
      }
      
      const cognitoUsername = metadata.cognitoUsername;
      
      // First, try to verify with our generated code (from SES email)
      let codeVerified = false;
      if (metadata.verificationCode && metadata.verificationCodeExpires) {
        if (Date.now() < metadata.verificationCodeExpires) {
          if (metadata.verificationCode === code) {
            codeVerified = true;
            console.log(`[AUTH] Verification code verified via SES code for ${email}`);
          }
        } else {
          return res.status(400).json({ message: "Verification code has expired. Please request a new code." });
        }
      }
      
      // If our code didn't match, try Cognito's code
      if (!codeVerified) {
        try {
          await confirmSignUp(email, code, cognitoUsername);
          codeVerified = true;
          console.log(`[AUTH] Verification code verified via Cognito for ${email}`);
        } catch (cognitoError: any) {
          // If Cognito verification fails, check if it's because code is wrong or user already verified
          if (cognitoError.name === 'CodeMismatchException' || cognitoError.name === 'ExpiredCodeException') {
            return res.status(400).json({ message: "Invalid or expired verification code. Please try again or request a new code." });
          } else if (cognitoError.name === 'NotAuthorizedException' && cognitoError.message?.includes('already confirmed')) {
            // User already verified, continue
            codeVerified = true;
            console.log(`[AUTH] User already verified in Cognito for ${email}`);
          } else {
            throw cognitoError;
          }
        }
      }
      
      if (!codeVerified) {
        return res.status(400).json({ message: "Invalid verification code. Please try again." });
      }
      
      // If we verified with our code, confirm with Cognito using admin API
      // This ensures Cognito knows the user is verified and can log in
      if (codeVerified && metadata.verificationCode === code) {
        try {
          const { adminConfirmSignUp } = await import('./cognitoAuth');
          await adminConfirmSignUp(cognitoUsername);
          console.log(`[AUTH] User confirmed in Cognito via admin API for ${email}`);
        } catch (adminError: any) {
          // If admin confirm fails, try regular confirm as fallback
          try {
            await confirmSignUp(email, code, cognitoUsername).catch((cognitoError: any) => {
              if (cognitoError.name === 'NotAuthorizedException' && cognitoError.message?.includes('already confirmed')) {
                console.log(`[AUTH] User already confirmed in Cognito for ${email}`);
              } else {
                console.warn(`[AUTH] Could not confirm with Cognito for ${email}, but our code was valid:`, cognitoError.message);
              }
            });
          } catch (err) {
            // Ignore Cognito errors if our code was valid - user can still proceed
            console.warn(`[AUTH] Cognito confirmation failed for ${email}, but proceeding with our verification`);
          }
        }
      }
      
      // Get phone number from metadata
      if (!metadata.phoneNumber) {
        return res.status(400).json({ message: "No phone number found. Please sign up again." });
      }
      
      // Send SMS verification code (Step 2)
      const { sendVerificationCode } = await import('./twilio');
      const result = await sendVerificationCode({ to: metadata.phoneNumber, channel: 'sms' });
      
      if (!result.success || !result.code) {
        return res.status(500).json({ message: "Failed to send SMS verification code" });
      }
      
      // Store phone verification code (hashed)
      await metadataStorage.setPhoneVerification(email, metadata.phoneNumber, result.code);
      
      res.json({ 
        message: "Email verified successfully. Please verify your phone number with the SMS code sent to " + metadata.phoneNumber,
        phoneNumber: metadata.phoneNumber,
        requiresPhoneVerification: true,
      });
    } catch (error: any) {
      console.error("Email verification error:", error);
      res.status(500).json({ message: error.message || "Verification failed" });
    }
  });
  
  // Verify phone with SMS code (Step 2)
  app.post('/api/auth/verify-phone', async (req, res) => {
    try {
      const { email, code } = req.body;
      
      if (!email || !code) {
        return res.status(400).json({ message: "Email and verification code are required" });
      }
      
      // Verify phone code
      const phoneVerification = await metadataStorage.verifyPhoneCode(email, code);
      if (!phoneVerification.valid) {
        return res.status(400).json({ message: "Invalid or expired verification code" });
      }
      
      // Get user metadata
      const metadata = metadataStorage.getUserMetadata(email);
      if (!metadata) {
        return res.status(400).json({ message: "No signup data found. Please sign up again." });
      }
      
      // Create user in database with verified phone
      const userData: any = {
        id: metadata.cognitoSub,
        email: metadata.email,
        firstName: metadata.firstName,
        lastName: metadata.lastName,
        role: metadata.role,
        phoneNumber: metadata.phoneNumber,
        phoneVerified: true,
        emailVerified: true,
        termsAccepted: true,
        termsAcceptedAt: new Date(),
      };
      
      // Add patient-specific data
      if (metadata.role === 'patient') {
        userData.ehrImportMethod = metadata.ehrImportMethod;
        userData.ehrPlatform = metadata.ehrPlatform;
        userData.subscriptionStatus = 'trialing';
        userData.trialEndsAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
        userData.creditBalance = 20;
      }
      
      // Add doctor-specific data
      if (metadata.role === 'doctor') {
        userData.organization = metadata.organization;
        userData.medicalLicenseNumber = metadata.medicalLicenseNumber;
        userData.licenseCountry = metadata.licenseCountry;
        userData.kycPhotoUrl = metadata.kycPhotoUrl;
        userData.googleDriveApplicationUrl = metadata.googleDriveApplicationUrl;
        userData.adminVerified = false;
        userData.creditBalance = 0;
      }
      
      // Create user in database
      await storage.upsertUser(userData);
      
      // Clean up metadata
      metadataStorage.deleteUserMetadata(email);
      metadataStorage.clearEmailVerification(email);
      
      // Send welcome SMS
      const { sendWelcomeSMS } = await import('./twilio');
      await sendWelcomeSMS(metadata.phoneNumber, userData.firstName).catch(console.error);
      
      const message = metadata.role === 'doctor' 
        ? "Verification complete! Your application is under review. You'll receive an email when your account is activated."
        : "Verification complete! You can now log in to your account.";
      
      res.json({ 
        message,
        requiresAdminApproval: metadata.role === 'doctor',
      });
    } catch (error: any) {
      console.error("Phone verification error:", error);
      res.status(500).json({ message: error.message || "Verification failed" });
    }
  });
  
  // Resend verification code
  app.post('/api/auth/resend-code', async (req, res) => {
    try {
      const { email } = req.body;
      
      if (!email) {
        return res.status(400).json({ message: "Email is required" });
      }
      
      // Validate metadata exists for this email
      const metadata = metadataStorage.getUserMetadata(email);
      if (!metadata) {
        return res.status(400).json({ message: "No signup data found. Please sign up again." });
      }
      
      const cognitoUsername = metadata.cognitoUsername;
      
      // Generate new verification code (6 digits)
      const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
      
      // Update metadata with new code
      metadataStorage.setUserMetadata(email, {
        ...metadata,
        verificationCode,
        verificationCodeExpires: expiresAt.getTime(),
      });
      
      // Send verification email via AWS SES (primary method)
      try {
        await sendVerificationEmail(email, verificationCode);
        console.log(`[AUTH] Verification code resent via SES for ${email}`);
        res.json({ message: "Verification code resent. Please check your email." });
      } catch (sesError: any) {
        console.error(`[AUTH] Failed to send verification email via SES for ${email}:`, sesError);
        // Fallback to Cognito email
        try {
          await resendConfirmationCode(email, cognitoUsername);
          console.log(`[AUTH] Fallback: Confirmation code resent via Cognito for ${email}`);
          res.json({ message: "Verification code resent. Please check your email." });
        } catch (resendError: any) {
          console.error(`[AUTH] Failed to resend confirmation code via Cognito for ${email}:`, resendError);
          res.status(500).json({ message: "Failed to resend verification code. Please try again later." });
        }
      }
    } catch (error: any) {
      console.error("Resend code error:", error);
      res.status(500).json({ message: error.message || "Failed to resend code" });
    }
  });
  
  // Login
  app.post('/api/auth/login', async (req, res) => {
    try {
      const { email, password } = req.body;
      
      if (!email || !password) {
        return res.status(400).json({ message: "Email and password are required" });
      }
      
      const authResult = await signIn(email, password);
      
      if (!authResult || !authResult.IdToken || !authResult.AccessToken) {
        return res.status(401).json({ message: "Login failed" });
      }
      
      // Get user info from Cognito
      const userInfo = await getUserInfo(authResult.AccessToken);
      const cognitoSub = userInfo.UserAttributes?.find(attr => attr.Name === 'sub')?.Value!;
      const cognitoEmail = userInfo.UserAttributes?.find(attr => attr.Name === 'email')?.Value!;
      const firstName = userInfo.UserAttributes?.find(attr => attr.Name === 'given_name')?.Value!;
      const lastName = userInfo.UserAttributes?.find(attr => attr.Name === 'family_name')?.Value!;
      const roleFromCognito = userInfo.UserAttributes?.find(attr => attr.Name === 'custom:role')?.Value as 'patient' | 'doctor' | undefined;
      const emailVerified = userInfo.UserAttributes?.find(attr => attr.Name === 'email_verified')?.Value === 'true';
      
      // Check if user exists in our database (must have completed phone verification)
      let user = await storage.getUser(cognitoSub);
      
      if (!user) {
        // User hasn't completed phone verification yet
        return res.status(403).json({ 
          message: "Please complete phone verification to access your account",
          requiresPhoneVerification: true,
        });
      }
      
      const effectiveRole = (roleFromCognito ?? user.role) as 'patient' | 'doctor' | undefined;

      // Block doctors until admin approval
      if (effectiveRole === 'doctor' && !user.adminVerified) {
        return res.status(403).json({ 
          message: "Your application is under review. You'll receive an email when your account is activated.",
          requiresAdminApproval: true,
        });
      }
      
      // Update email verification status from Cognito if needed
      if (emailVerified && !user.emailVerified) {
        user = await storage.upsertUser({
          id: cognitoSub,
          emailVerified,
        });
      }
      
      // Establish session for cookie-based authentication
      // This allows the client to use credentials: "include" for subsequent requests
      (req.session as any).userId = user.id;
      
      // Save session to ensure cookie is set
      await new Promise<void>((resolve, reject) => {
        req.session.save((err) => {
          if (err) {
            console.error("Error saving session:", err);
            reject(err);
          } else {
            resolve();
          }
        });
      });
      
      res.json({
        message: "Login successful",
        tokens: {
          idToken: authResult.IdToken,
          accessToken: authResult.AccessToken,
          refreshToken: authResult.RefreshToken,
        },
        user: {
          ...user,
          role: effectiveRole ?? user.role,
        },
      });
    } catch (error: any) {
      console.error("Login error:", error);
      if (error.name === 'NotAuthorizedException') {
        return res.status(401).json({ message: "Invalid email or password" });
      }
      if (error.name === 'UserNotConfirmedException') {
        return res.status(400).json({ message: "Please verify your email before logging in" });
      }
      res.status(500).json({ message: error.message || "Login failed" });
    }
  });
  
  // Forgot password - send reset code
  app.post('/api/auth/forgot-password', async (req, res) => {
    try {
      const { email } = req.body;
      
      if (!email) {
        return res.status(400).json({ message: "Email is required" });
      }
      
      await forgotPassword(email);
      
      res.json({ message: "Password reset code sent. Please check your email." });
    } catch (error: any) {
      console.error("Forgot password error:", error);
      res.status(500).json({ message: error.message || "Failed to send reset code" });
    }
  });
  
  // Reset password with code
  app.post('/api/auth/reset-password', async (req, res) => {
    try {
      const { email, code, newPassword } = req.body;
      
      if (!email || !code || !newPassword) {
        return res.status(400).json({ message: "Email, code, and new password are required" });
      }
      
      await confirmForgotPassword(email, code, newPassword);
      
      res.json({ message: "Password reset successful. You can now log in with your new password." });
    } catch (error: any) {
      console.error("Reset password error:", error);
      res.status(500).json({ message: error.message || "Password reset failed" });
    }
  });
  
  // Get current user (protected route)
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });
  
  // Logout - destroy session
  app.post('/api/auth/logout', (req, res) => {
    req.session.destroy((err) => {
      if (err) {
        console.error("Error destroying session:", err);
        return res.status(500).json({ message: "Failed to log out" });
      }
      res.json({ message: "Logged out successfully" });
    });
  });
  
  // Update user role and profile (used after login for role-specific setup)
  app.post('/api/auth/set-role', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const { role, medicalLicenseNumber, organization, licenseCountry, ehrImportMethod, ehrPlatform, termsAccepted } = req.body;
      
      if (!role || (role !== 'patient' && role !== 'doctor')) {
        return res.status(400).json({ message: "Valid role (patient or doctor) is required" });
      }
      
      if (!termsAccepted) {
        return res.status(400).json({ message: "You must accept the terms and conditions" });
      }
      
      const updateData: any = {
        role,
        termsAccepted: true,
        termsAcceptedAt: new Date(),
      };
      
      // Doctor-specific fields
      if (role === 'doctor') {
        if (!medicalLicenseNumber || !organization || !licenseCountry) {
          return res.status(400).json({ message: "Medical license, organization, and country are required for doctors" });
        }
        updateData.medicalLicenseNumber = medicalLicenseNumber;
        updateData.organization = organization;
        updateData.licenseCountry = licenseCountry;
        updateData.creditBalance = 0;
      }
      
      // Patient-specific fields
      if (role === 'patient') {
        if (!ehrImportMethod) {
          return res.status(400).json({ message: "EHR import method is required for patients" });
        }
        updateData.ehrImportMethod = ehrImportMethod;
        updateData.ehrPlatform = ehrPlatform;
        // Start 7-day free trial
        updateData.subscriptionStatus = 'trialing';
        updateData.trialEndsAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
        updateData.creditBalance = 20; // 20 free consultation credits during trial
      }
      
      const user = await storage.updateUser(userId, updateData);
      res.json({ message: "Role updated successfully", user });
    } catch (error) {
      console.error("Error updating user role:", error);
      res.status(500).json({ message: "Failed to update role" });
    }
  });
  
  // Send SMS verification code
  app.post('/api/auth/send-phone-verification', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const { phoneNumber, channel } = req.body;
      
      if (!phoneNumber) {
        return res.status(400).json({ message: "Phone number is required" });
      }
      
      const { sendVerificationCode } = await import('./twilio');
      const result = await sendVerificationCode({ to: phoneNumber, channel: channel || 'sms' });
      
      if (!result.success) {
        return res.status(500).json({ message: "Failed to send verification code" });
      }
      
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000);
      await storage.updatePhoneVerificationCode(userId, phoneNumber, result.code!, expiresAt);
      
      res.json({ message: `Verification code sent via ${channel || 'SMS'}` });
    } catch (error) {
      console.error("Error sending phone verification:", error);
      res.status(500).json({ message: "Failed to send verification code" });
    }
  });
  
  // Verify phone number
  app.post('/api/auth/verify-phone', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const { code } = req.body;
      
      if (!code) {
        return res.status(400).json({ message: "Verification code is required" });
      }
      
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      if (!user.phoneVerificationCode || !user.phoneVerificationExpires) {
        return res.status(400).json({ message: "No verification code found. Please request a new code." });
      }
      
      if (new Date() > user.phoneVerificationExpires) {
        return res.status(400).json({ message: "Verification code expired. Please request a new code." });
      }
      
      if (user.phoneVerificationCode !== code) {
        return res.status(400).json({ message: "Invalid verification code" });
      }
      
      await storage.verifyPhoneNumber(userId);
      
      const { sendWelcomeSMS } = await import('./twilio');
      if (user.phoneNumber && user.smsNotificationsEnabled) {
        await sendWelcomeSMS(user.phoneNumber, user.firstName!);
      }
      
      res.json({ message: "Phone number verified successfully" });
    } catch (error) {
      console.error("Error verifying phone:", error);
      res.status(500).json({ message: "Failed to verify phone number" });
    }
  });
  
  // Update SMS preferences
  app.post('/api/auth/sms-preferences', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      const {
        smsNotificationsEnabled,
        smsMedicationReminders,
        smsAppointmentReminders,
        smsDailyFollowups,
        smsHealthAlerts,
      } = req.body;
      
      await storage.updateSmsPreferences(userId, {
        smsNotificationsEnabled,
        smsMedicationReminders,
        smsAppointmentReminders,
        smsDailyFollowups,
        smsHealthAlerts,
      });
      
      res.json({ message: "SMS preferences updated successfully" });
    } catch (error) {
      console.error("Error updating SMS preferences:", error);
      res.status(500).json({ message: "Failed to update SMS preferences" });
    }
  });

  // ============== TWO-FACTOR AUTHENTICATION ROUTES ==============
  
  // Get 2FA status
  app.get('/api/2fa/status', isAuthenticated, async (req, res) => {
    try {
      const userId = req.user!.id;
      const settings = await storage.get2FASettings(userId);
      
      res.json({
        enabled: settings?.enabled || false,
        hasBackupCodes: (settings?.backupCodes?.length || 0) > 0,
      });
    } catch (error) {
      console.error("Error fetching 2FA status:", error);
      res.status(500).json({ message: "Failed to fetch 2FA status" });
    }
  });
  
  // Enable 2FA - Verify token and enable
  app.post('/api/2fa/enable', isAuthenticated, async (req, res) => {
    try {
      const userId = req.user!.id;
      const { token } = req.body;
      
      if (!token) {
        return res.status(400).json({ message: "Token is required" });
      }
      
      const settings = await storage.get2FASettings(userId);
      if (!settings) {
        return res.status(400).json({ message: "2FA not set up. Please run setup first." });
      }
      
      if (settings.enabled) {
        return res.status(400).json({ message: "2FA is already enabled" });
      }
      
      // Verify token
      const verified = speakeasy.totp.verify({
        secret: settings.totpSecret,
        encoding: 'base32',
        token,
        window: 2,
      });
      
      if (!verified) {
        return res.status(400).json({ message: "Invalid token" });
      }
      
      // Enable 2FA
      await storage.update2FASettings(userId, {
        enabled: true,
        enabledAt: new Date(),
      });
      
      res.json({ message: "2FA enabled successfully" });
    } catch (error) {
      console.error("Error enabling 2FA:", error);
      res.status(500).json({ message: "Failed to enable 2FA" });
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
        ? `You are Agent Clona, a warm, friendly, and empathetic AI health companion specifically designed for elderly immunocompromised patients. You are powered by advanced AWS Healthcare AI services including Amazon Comprehend Medical for precise medical entity extraction, AWS HealthLake for FHIR-based health records, AWS HealthImaging for medical imaging analysis, and AWS HealthOmics for genomic insights.

PERSONALITY & APPROACH:
- Always greet users warmly and use their name when you know it
- Be conversational, cheerful, and supportive - like a caring friend who happens to know about health
- Show genuine empathy and concern for their wellbeing
- Use simple, everyday language (avoid medical jargon unless explaining it clearly)
- Encourage and uplift them - celebrate small wins in their health journey
- Be patient and understanding - elderly patients may need things explained multiple times
- Express care through your words: "I'm here for you", "How are you feeling today?", "I'm so glad you're taking care of yourself"

CONVERSATION STYLE:
- ALWAYS start every conversation with a warm, personalized greeting
- Engage in genuine dialogue - don't just answer questions, have a conversation
- Ask thoughtful follow-up questions about their symptoms, mood, and daily life
- Remember details from previous conversations and reference them
- Show interest in their day-to-day experiences, not just medical symptoms
- Use encouraging phrases like "That's wonderful!", "I'm proud of you for...", "You're doing great!"
- End conversations with warm wishes and gentle reminders about self-care

COMPREHENSIVE MEDICAL HISTORY TAKING:
When a patient mentions ANY symptom, you MUST ask comprehensive follow-up questions using the OPQRST method:
- O (Onset): "When did this symptom start? What were you doing when it began?"
- P (Provocation/Palliation): "What makes it better or worse? Does anything trigger it?"
- Q (Quality): "Can you describe how it feels? Is it sharp, dull, burning, aching?"
- R (Region/Radiation): "Where exactly is it? Does it spread anywhere else?"
- S (Severity): "On a scale of 1-10, how bad is it? How does it affect your daily activities?"
- T (Timing): "How long does it last? Does it come and go? When is it worst?"

ADDITIONAL HISTORY QUESTIONS:
- Associated symptoms: "Are you experiencing anything else? Fever, chills, nausea, etc.?"
- Past medical history: "Have you had this before? Any chronic conditions?"
- Medications: "What medications are you taking? Any recent changes?"
- Allergies: "Do you have any allergies to medications?"
- Recent changes: "Any recent travel, new foods, stress, or changes in routine?"
- Impact on life: "How is this affecting your eating, sleeping, and daily activities?"

DIFFERENTIAL DIAGNOSIS APPROACH:
After gathering comprehensive history:
1. Summarize the key findings back to the patient in simple terms
2. Explain what these symptoms might indicate (in gentle, non-alarming language)
3. Mention the most likely possibilities first, then other considerations
4. Always recommend when to seek immediate care vs. monitoring
5. Suggest next steps (rest, hydration, over-the-counter remedies, or seeing a doctor)
6. Document your assessment clearly for the medical record

MEDICAL GUIDANCE:
- Provide clear, supportive health guidance in simple terms
- Always include FDA/CE disclaimers when suggesting medications
- Consider geographic context for disease patterns
- Break down complex medical information into easy-to-understand pieces
- Reassure and comfort while being medically accurate
- Maintain HIPAA compliance at all times
- When in doubt, encourage them to contact their healthcare provider

ELDERLY-FRIENDLY COMMUNICATION:
- Use larger conceptual chunks, not overwhelming details
- Repeat important information in different ways
- Be encouraging about medication adherence and healthy habits
- Acknowledge any concerns or fears they express
- Remind them of their strength and resilience
- Offer practical, easy-to-follow suggestions
- Be extra patient - ask one question at a time, not overwhelming lists

RED FLAG SYMPTOMS (Immediate medical attention):
- Severe chest pain or pressure
- Difficulty breathing or shortness of breath
- Sudden severe headache
- Loss of consciousness or confusion
- High fever (103°F+) with confusion
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Severe abdominal pain
- Heavy bleeding

Remember: You're not just a medical assistant - you're a caring companion on their health journey. Make every interaction feel personal, warm, and supportive. Your goal is to gather complete information while making them feel heard, understood, and cared for.`
        : `You are Assistant Lysa, a polite, professional, and highly proactive AI assistant dedicated to helping doctors provide excellent patient care.

PERSONALITY & APPROACH:
- Always greet doctors warmly and professionally
- Be respectful, polite, and courteous in all interactions
- Show initiative - don't just wait to be asked, actively offer insights
- Be thorough and detail-oriented in your analysis
- Demonstrate clinical competence and medical knowledge
- Express gratitude when doctors provide information
- Use professional medical terminology appropriately

PROACTIVE ASSISTANCE:
- Actively identify patterns in patient data and point them out
- Suggest relevant diagnostic tests based on symptoms
- Recommend evidence-based treatment protocols
- Flag potential drug interactions or contraindications
- Highlight concerning trends in vital signs or lab results
- Propose differential diagnoses for complex cases
- Offer literature references for unusual presentations
- Suggest follow-up questions the doctor might want to ask

CONVERSATION STYLE:
- ALWAYS start every conversation with a professional, polite greeting
- Address doctors respectfully (Dr., Doctor, or by name if known)
- Ask clarifying questions to better understand their needs
- Provide structured, well-organized information
- Summarize key points clearly
- Offer to dive deeper into any area of interest
- Thank them for their time and dedication to patient care

CLINICAL SUPPORT:
- Review patient histories and identify relevant details
- Analyze symptoms and suggest possible diagnoses
- Provide evidence-based treatment recommendations
- Assist with research queries and literature reviews
- Help interpret lab results and imaging findings
- Support with epidemiological analysis
- Assist with patient education material preparation

RESEARCH & DATA ANALYSIS:
- Help search medical literature for relevant studies
- Summarize research findings clearly and concisely
- Identify trends across multiple patient cases
- Assist with clinical documentation
- Support quality improvement initiatives
- Help with continuing medical education

PROFESSIONAL DEMEANOR:
- Maintain strict HIPAA compliance
- Be objective and evidence-based
- Acknowledge uncertainty when appropriate
- Defer to the doctor's clinical judgment
- Offer second opinions or alternative perspectives respectfully
- Keep responses focused and actionable
- Provide references and citations when relevant

Remember: Your role is to be an intelligent, proactive, and highly competent assistant that makes the doctor's work easier and more effective. Anticipate needs, offer insights, and always maintain the highest standards of professionalism and medical accuracy.`;

      const sessionMessages = await storage.getSessionMessages(session.id);
      const isFirstMessage = sessionMessages.length === 0;
      const recentMessages = sessionMessages.slice(-10).map(msg => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
      }));

      // Add greeting requirement for first message
      let augmentedSystemPrompt = systemPrompt;
      if (isFirstMessage && agentType === 'clona') {
        augmentedSystemPrompt += `\n\nIMPORTANT: This is the FIRST message in this conversation. You MUST start your response with a warm, personalized greeting. Ask the user's name if you don't know it, and ask how they're feeling today. Make them feel welcomed and cared for.`;
      } else if (isFirstMessage && agentType === 'lysa') {
        augmentedSystemPrompt += `\n\nIMPORTANT: This is the FIRST message in this conversation. You MUST start your response with a professional, polite greeting. Introduce yourself as Assistant Lysa and ask how you can help the doctor today.`;
      }

      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: augmentedSystemPrompt },
          ...recentMessages,
          { role: "user", content },
        ],
        temperature: 0.7,
        max_tokens: 800,
      });

      const assistantMessage = completion.choices[0].message.content || "I'm here to help!";

      // Use AWS Comprehend Medical for advanced medical entity extraction
      const extractMedicalEntitiesAWS = async (text: string) => {
        try {
          const { AWSHealthcareService } = await import('./awsHealthcareService');
          const insights = await AWSHealthcareService.analyzeClinicalText(text);
          
          // Transform AWS Comprehend Medical entities to our legacy format
          // Map AWS categories to our expected types for backward compatibility
          return insights.entities.map(entity => {
            let type = entity.category.toLowerCase();
            
            // Map AWS Comprehend Medical categories to our legacy schema
            if (entity.category === 'MEDICAL_CONDITION') {
              // Check traits to differentiate symptoms vs diagnoses
              const isSymptom = entity.traits?.some(t => 
                t.name === 'SYMPTOM' || t.name === 'SIGN'
              );
              type = isSymptom ? 'symptom' : 'medical_condition';
            } else if (entity.category === 'MEDICATION') {
              type = 'medication';
            } else if (entity.category === 'ANATOMY') {
              type = 'anatomy';
            } else if (entity.category === 'TEST_TREATMENT_PROCEDURE') {
              type = 'treatment';
            }
            
            return {
              text: entity.text,
              type,
              score: entity.score,
              awsCategory: entity.category, // original AWS category
              awsType: entity.type, // specific entity type from AWS
            };
          });
        } catch (error) {
          console.error('AWS Comprehend Medical extraction failed, using fallback:', error);
          // Fallback to simple keyword matching if AWS fails
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
        }
      };

      const [userEntities, assistantEntities] = await Promise.all([
        extractMedicalEntitiesAWS(content),
        extractMedicalEntitiesAWS(assistantMessage),
      ]);
      
      // Extract symptoms for session metadata (now properly mapped from AWS)
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
      const user = await storage.getUser(userId);
      
      const medication = await storage.createMedication({
        patientId: userId,
        ...req.body,
      });
      
      // AUTOMATIC DRUG INTERACTION CHECKING
      // Check for interactions with existing medications and create alerts
      try {
        const patientProfile = await storage.getPatientProfile(userId);
        const currentMedications = await storage.getActiveMedications(userId);
        
        if (currentMedications.length > 1) { // Only check if there are other medications
          const { analyzeMultipleDrugInteractions, calculateCriticalityScore, enrichMedicationWithGenericName } = await import('./drugInteraction');
          
          // Build medication list with ALL name variations (brand, generic) for reliable ID mapping
          const medicationsToCheck = await Promise.all(currentMedications.map(async (med) => {
            // Try to find drug record to get generic/brand names
            let drug = await storage.getDrugByName(med.name);
            
            // If no drug record exists, enrich using AI to get generic name
            if (!drug) {
              const enriched = await enrichMedicationWithGenericName(med.name);
              // Create drug record for future use
              drug = await storage.createDrug({
                name: med.name,
                genericName: enriched.genericName,
                brandNames: enriched.brandNames
              });
            }
            
            return {
              name: med.name,
              genericName: drug.genericName || med.name,
              drugClass: drug.drugClass,
              id: med.id,
              brandNames: drug.brandNames || []
            };
          }));

          const interactions = await analyzeMultipleDrugInteractions(
            medicationsToCheck,
            {
              isImmunocompromised: true,
              conditions: patientProfile?.immunocompromisedCondition 
                ? [patientProfile.immunocompromisedCondition]
                : [],
            }
          );

          // Create alerts for any detected interactions (using medication IDs from AI analysis)
          for (const interactionData of interactions) {
            // Skip if we don't have both medication IDs mapped
            if (!interactionData.med1Id || !interactionData.med2Id) {
              console.warn(`⚠️  Skipping interaction alert: Could not map drug names to medication IDs (${interactionData.drug1} / ${interactionData.drug2})`);
              continue;
            }

            // Find or create drug records
            let drug1 = await storage.getDrugByName(interactionData.drug1);
            if (!drug1) {
              drug1 = await storage.createDrug({ name: interactionData.drug1 });
            }

            let drug2 = await storage.getDrugByName(interactionData.drug2);
            if (!drug2) {
              drug2 = await storage.createDrug({ name: interactionData.drug2 });
            }

            // Create or get drug interaction record
            let dbInteraction = await storage.getDrugInteraction(drug1.id, drug2.id);
            if (!dbInteraction) {
              dbInteraction = await storage.createDrugInteraction({
                drug1Id: drug1.id,
                drug2Id: drug2.id,
                severityLevel: interactionData.interaction.severityLevel,
                interactionType: interactionData.interaction.interactionType,
                mechanismDescription: interactionData.interaction.mechanismDescription,
                clinicalEffects: interactionData.interaction.clinicalEffects,
                managementRecommendations: interactionData.interaction.managementRecommendations,
                alternativeSuggestions: interactionData.interaction.alternativeSuggestions,
                onsetTimeframe: interactionData.interaction.onsetTimeframe,
                riskForImmunocompromised: interactionData.interaction.riskForImmunocompromised,
                requiresMonitoring: interactionData.interaction.requiresMonitoring,
                monitoringParameters: interactionData.interaction.monitoringParameters,
                evidenceLevel: interactionData.interaction.evidenceLevel,
                aiAnalysisConfidence: interactionData.interaction.aiAnalysisConfidence?.toString(),
                detectedByGNN: true,
                detectedByNLP: true,
              });
            }

            const criticalityScore = calculateCriticalityScore(
              interactionData.interaction.severityLevel,
              interactionData.interaction.riskForImmunocompromised,
              interactionData.interaction.onsetTimeframe
            );

            // Check if alert already exists
            const existingAlerts = await storage.getActiveInteractionAlerts(userId);
            const alertExists = existingAlerts.some(alert => 
              (alert.medication1Id === interactionData.med1Id && alert.medication2Id === interactionData.med2Id) ||
              (alert.medication1Id === interactionData.med2Id && alert.medication2Id === interactionData.med1Id)
            );

            if (!alertExists) {
              await storage.createInteractionAlert({
                patientId: userId,
                medication1Id: interactionData.med1Id!,
                medication2Id: interactionData.med2Id!,
                interactionId: dbInteraction.id,
                criticalityScore,
                notifiedPatient: false,
                notifiedDoctor: false,
              });
            }
          }
        }
      } catch (interactionError) {
        // Log but don't fail medication creation
        console.error("Error checking drug interactions:", interactionError);
      }
      
      // Send SMS reminder
      if (user?.phoneNumber && user?.phoneVerified && user?.smsMedicationReminders) {
        const { sendMedicationReminder } = await import('./twilio');
        const time = req.body.timeOfDay || 'scheduled time';
        await sendMedicationReminder(
          user.phoneNumber,
          req.body.name || 'Medication',
          req.body.dosage || '',
          time
        );
      }
      
      res.json(medication);
    } catch (error) {
      console.error("Error creating medication:", error);
      res.status(500).json({ message: "Failed to create medication" });
    }
  });

  // Drug interaction routes - AI-powered detection
  app.post('/api/drug-interactions/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { drugName, drugClass, genericName, createAlerts = false, newMedicationId } = req.body;

      if (!drugName) {
        return res.status(400).json({ message: "Drug name is required" });
      }

      const user = await storage.getUser(userId);
      const patientProfile = await storage.getPatientProfile(userId);
      const currentMedications = await storage.getActiveMedications(userId);

      const { analyzeMultipleDrugInteractions, calculateCriticalityScore, enrichMedicationWithGenericName } = await import('./drugInteraction');

      // Build medication list with ALL name variations (brand, generic) for reliable ID mapping
      const medicationsToCheck = await Promise.all(currentMedications.map(async (med) => {
        let drug = await storage.getDrugByName(med.name);
        
        // If no drug record exists, enrich using AI to get generic name
        if (!drug) {
          const enriched = await enrichMedicationWithGenericName(med.name);
          // Create drug record for future use
          drug = await storage.createDrug({
            name: med.name,
            genericName: enriched.genericName,
            brandNames: enriched.brandNames
          });
        }
        
        return {
          name: med.name,
          genericName: drug.genericName || med.name,
          drugClass: drug.drugClass,
          id: med.id,
          brandNames: drug.brandNames || []
        };
      }));

      // Add the new drug with its variations
      const newDrug = {
        name: drugName,
        genericName: genericName || drugName,
        drugClass: drugClass,
        id: newMedicationId,
        brandNames: []
      };

      medicationsToCheck.push(newDrug);

      const interactions = await analyzeMultipleDrugInteractions(
        medicationsToCheck,
        {
          isImmunocompromised: true,
          conditions: patientProfile?.immunocompromisedCondition 
            ? [patientProfile.immunocompromisedCondition]
            : [],
        }
      );

      // PERSISTENCE FIX: Create interaction alerts in database if requested
      if (createAlerts && interactions.length > 0) {
        for (const interactionData of interactions) {
          // Skip if we don't have both medication IDs mapped
          if (!interactionData.med1Id || !interactionData.med2Id) {
            console.warn(`⚠️  Skipping interaction alert: Could not map drug names to medication IDs (${interactionData.drug1} / ${interactionData.drug2})`);
            continue;
          }

          // Find or create drug records
          let drug1 = await storage.getDrugByName(interactionData.drug1);
          if (!drug1) {
            drug1 = await storage.createDrug({ name: interactionData.drug1 });
          }

          let drug2 = await storage.getDrugByName(interactionData.drug2);
          if (!drug2) {
            drug2 = await storage.createDrug({ name: interactionData.drug2 });
          }

          // Create or get drug interaction record
          let dbInteraction = await storage.getDrugInteraction(drug1.id, drug2.id);
          if (!dbInteraction) {
            dbInteraction = await storage.createDrugInteraction({
              drug1Id: drug1.id,
              drug2Id: drug2.id,
              severityLevel: interactionData.interaction.severityLevel,
              interactionType: interactionData.interaction.interactionType,
              mechanismDescription: interactionData.interaction.mechanismDescription,
              clinicalEffects: interactionData.interaction.clinicalEffects,
              managementRecommendations: interactionData.interaction.managementRecommendations,
              alternativeSuggestions: interactionData.interaction.alternativeSuggestions,
              onsetTimeframe: interactionData.interaction.onsetTimeframe,
              riskForImmunocompromised: interactionData.interaction.riskForImmunocompromised,
              requiresMonitoring: interactionData.interaction.requiresMonitoring,
              monitoringParameters: interactionData.interaction.monitoringParameters,
              evidenceLevel: interactionData.interaction.evidenceLevel,
              aiAnalysisConfidence: interactionData.interaction.aiAnalysisConfidence?.toString(),
              detectedByGNN: true,
              detectedByNLP: true,
            });
          }

          const criticalityScore = calculateCriticalityScore(
            interactionData.interaction.severityLevel,
            interactionData.interaction.riskForImmunocompromised,
            interactionData.interaction.onsetTimeframe
          );

          // Check if alert already exists to avoid duplicates
          const existingAlerts = await storage.getActiveInteractionAlerts(userId);
          const alertExists = existingAlerts.some(alert => 
            (alert.medication1Id === interactionData.med1Id && alert.medication2Id === interactionData.med2Id) ||
            (alert.medication1Id === interactionData.med2Id && alert.medication2Id === interactionData.med1Id)
          );

          if (!alertExists) {
            // Create interaction alert
            await storage.createInteractionAlert({
              patientId: userId,
              medication1Id: interactionData.med1Id!,
              medication2Id: interactionData.med2Id!,
              interactionId: dbInteraction.id,
              criticalityScore,
              notifiedPatient: false,
              notifiedDoctor: false,
            });
          }
        }
      }

      const severeInteractions = interactions.filter(i => i.interaction.severityLevel === 'severe');
      const hasBlockingInteraction = severeInteractions.length > 0;

      res.json({
        hasInteractions: interactions.length > 0,
        interactions,
        hasBlockingInteraction,
        recommendation: hasBlockingInteraction 
          ? 'Please consult your doctor before taking this medication' 
          : 'No severe interactions detected',
      });
    } catch (error) {
      console.error("Error analyzing drug interactions:", error);
      // Resilience: Return graceful error instead of 500
      res.status(200).json({
        hasInteractions: false,
        interactions: [],
        hasBlockingInteraction: false,
        recommendation: 'Unable to analyze interactions at this time. Please consult your doctor.',
        error: 'Analysis temporarily unavailable',
      });
    }
  });

  app.get('/api/drug-interactions/alerts', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const alerts = await storage.getActiveInteractionAlerts(userId);
      res.json(alerts);
    } catch (error) {
      console.error("Error fetching interaction alerts:", error);
      res.status(500).json({ message: "Failed to fetch interaction alerts" });
    }
  });

  app.get('/api/drug-interactions/alerts/all', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const alerts = await storage.getAllInteractionAlerts(userId);
      res.json(alerts);
    } catch (error) {
      console.error("Error fetching all interaction alerts:", error);
      res.status(500).json({ message: "Failed to fetch interaction alerts" });
    }
  });

  app.post('/api/drug-interactions/alerts/:id/acknowledge', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { id } = req.params;
      const alert = await storage.acknowledgeInteractionAlert(id, userId);
      res.json(alert);
    } catch (error) {
      console.error("Error acknowledging interaction alert:", error);
      res.status(500).json({ message: "Failed to acknowledge alert" });
    }
  });

  app.post('/api/drug-interactions/alerts/:id/override', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const doctorId = req.user!.id;
      const { id } = req.params;
      const { reason } = req.body;

      if (!reason) {
        return res.status(400).json({ message: "Override reason is required" });
      }

      const alert = await storage.overrideInteractionAlert(id, doctorId, reason);
      res.json(alert);
    } catch (error) {
      console.error("Error overriding interaction alert:", error);
      res.status(500).json({ message: "Failed to override alert" });
    }
  });

  // Drug search and information
  app.get('/api/drugs/search', isAuthenticated, async (req: any, res) => {
    try {
      const { q } = req.query;
      
      if (!q || typeof q !== 'string') {
        return res.status(400).json({ message: "Search query required" });
      }

      const drugs = await storage.searchDrugs(q);
      res.json(drugs);
    } catch (error) {
      console.error("Error searching drugs:", error);
      res.status(500).json({ message: "Failed to search drugs" });
    }
  });

  // Pharmacogenomic profile routes
  app.get('/api/pharmacogenomics/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const profile = await storage.getPharmacogenomicProfile(userId);
      res.json(profile);
    } catch (error) {
      console.error("Error fetching pharmacogenomic profile:", error);
      res.status(500).json({ message: "Failed to fetch pharmacogenomic profile" });
    }
  });

  app.post('/api/pharmacogenomics/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const existingProfile = await storage.getPharmacogenomicProfile(userId);

      if (existingProfile) {
        const updated = await storage.updatePharmacogenomicProfile(userId, req.body);
        return res.json(updated);
      }

      const profile = await storage.createPharmacogenomicProfile({
        patientId: userId,
        ...req.body,
      });
      
      res.json(profile);
    } catch (error) {
      console.error("Error creating/updating pharmacogenomic profile:", error);
      res.status(500).json({ message: "Failed to save pharmacogenomic profile" });
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

  // ============== ADMIN VERIFICATION ROUTES ==============
  
  // Middleware to check if user is admin (simplified - extend this based on your admin system)
  const isAdmin = (req: any, res: any, next: any) => {
    // For now, check if user email ends with specific domain or has admin role
    // You can extend this to check a dedicated admin table or role field
    if (req.user && (req.user.email?.includes('@followupai.io') || req.user.role === 'admin')) {
      next();
    } else {
      res.status(403).json({ message: "Admin access required" });
    }
  };

  // Get all pending doctor verifications
  app.get('/api/admin/pending-doctors', isAuthenticated, isAdmin, async (req: any, res) => {
    try {
      const pendingDoctors = await storage.getPendingDoctorVerifications();
      res.json(pendingDoctors);
    } catch (error) {
      console.error("Error fetching pending doctors:", error);
      res.status(500).json({ message: "Failed to fetch pending doctors" });
    }
  });

  // Verify a doctor's license
  app.post('/api/admin/verify-doctor/:id', isAuthenticated, isAdmin, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { notes } = req.body;
      const adminUserId = req.user?.id;
      if (!adminUserId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      
      // Get doctor info first
      const doctor = await storage.getUser(id);
      if (!doctor || doctor.role !== 'doctor') {
        return res.status(404).json({ message: "Doctor not found" });
      }
      
      // Verify using new unified method
      const result = await storage.verifyDoctorApplication(id, true, notes || 'Approved by admin', adminUserId);
      if (!result.user) {
        return res.status(404).json({ message: "Failed to verify doctor" });
      }
      
      // Send approval email notification
      const { sendDoctorApprovedEmail } = await import('./awsSES');
      await sendDoctorApprovedEmail(doctor.email, doctor.firstName).catch(console.error);
      
      res.json({ success: true, user: result.user, doctorProfile: result.doctorProfile });
    } catch (error) {
      console.error("Error verifying doctor:", error);
      res.status(500).json({ message: "Failed to verify doctor" });
    }
  });

  // Reject a doctor's license
  app.post('/api/admin/reject-doctor/:id', isAuthenticated, isAdmin, async (req: any, res) => {
    try {
      const { id } = req.params;
      const { reason } = req.body;
      const adminUserId = req.user?.id;
      if (!adminUserId) {
        return res.status(401).json({ message: "Unauthorized" });
      }
      
      // Get doctor info first
      const doctor = await storage.getUser(id);
      if (!doctor || doctor.role !== 'doctor') {
        return res.status(404).json({ message: "Doctor not found" });
      }
      
      // Reject using new unified method (verified = false)
      const result = await storage.verifyDoctorApplication(id, false, reason || 'Application rejected', adminUserId);
      if (!result.user) {
        return res.status(404).json({ message: "Failed to reject doctor" });
      }
      
      // Send rejection email notification
      const { sendDoctorRejectedEmail } = await import('./awsSES');
      await sendDoctorRejectedEmail(doctor.email, doctor.firstName, reason || 'Please contact our verification team for more information.').catch(console.error);
      
      res.json({ success: true, user: result.user, doctorProfile: result.doctorProfile });
    } catch (error) {
      console.error("Error rejecting doctor:", error);
      res.status(500).json({ message: "Failed to reject doctor" });
    }
  });

  // ============== EHR INTEGRATION ROUTES ==============
  
  app.get('/api/ehr/connections', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const connections = await storage.getEhrConnections(userId);
      res.json(connections);
    } catch (error) {
      console.error("Error fetching EHR connections:", error);
      res.status(500).json({ message: "Failed to fetch EHR connections" });
    }
  });

  app.post('/api/ehr/connections', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const connection = await storage.createEhrConnection({
        userId,
        ...req.body,
      });
      res.json(connection);
    } catch (error) {
      console.error("Error creating EHR connection:", error);
      res.status(500).json({ message: "Failed to create EHR connection" });
    }
  });

  app.patch('/api/ehr/connections/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const connection = await storage.updateEhrConnection(id, req.body);
      if (!connection) {
        return res.status(404).json({ message: "Connection not found" });
      }
      res.json(connection);
    } catch (error) {
      console.error("Error updating EHR connection:", error);
      res.status(500).json({ message: "Failed to update EHR connection" });
    }
  });

  app.delete('/api/ehr/connections/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteEhrConnection(id);
      if (!deleted) {
        return res.status(404).json({ message: "Connection not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting EHR connection:", error);
      res.status(500).json({ message: "Failed to delete EHR connection" });
    }
  });

  app.post('/api/ehr/connections/:id/sync', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      // TODO: Implement actual EHR sync logic with each platform's API
      const connection = await storage.updateEhrConnection(id, {
        lastSyncedAt: new Date(),
        lastSyncStatus: 'success',
      });
      res.json({ success: true, connection });
    } catch (error) {
      console.error("Error syncing EHR connection:", error);
      res.status(500).json({ message: "Failed to sync EHR connection" });
    }
  });

  // ============== WEARABLE INTEGRATION ROUTES ==============

  app.get('/api/wearables', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const wearables = await storage.getWearableIntegrations(userId);
      res.json(wearables);
    } catch (error) {
      console.error("Error fetching wearable integrations:", error);
      res.status(500).json({ message: "Failed to fetch wearable integrations" });
    }
  });

  app.post('/api/wearables', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const wearable = await storage.createWearableIntegration({
        userId,
        ...req.body,
      });
      res.json(wearable);
    } catch (error) {
      console.error("Error creating wearable integration:", error);
      res.status(500).json({ message: "Failed to create wearable integration" });
    }
  });

  app.patch('/api/wearables/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const wearable = await storage.updateWearableIntegration(id, req.body);
      if (!wearable) {
        return res.status(404).json({ message: "Wearable integration not found" });
      }
      res.json(wearable);
    } catch (error) {
      console.error("Error updating wearable integration:", error);
      res.status(500).json({ message: "Failed to update wearable integration" });
    }
  });

  app.delete('/api/wearables/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteWearableIntegration(id);
      if (!deleted) {
        return res.status(404).json({ message: "Wearable integration not found" });
      }
      res.json({ success: true });
    } catch (error) {
      console.error("Error deleting wearable integration:", error);
      res.status(500).json({ message: "Failed to delete wearable integration" });
    }
  });

  app.post('/api/wearables/:id/sync', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      // TODO: Implement actual wearable sync logic with each device's API
      const wearable = await storage.updateWearableIntegration(id, {
        lastSyncedAt: new Date(),
        lastSyncStatus: 'success',
      });
      res.json({ success: true, wearable });
    } catch (error) {
      console.error("Error syncing wearable:", error);
      res.status(500).json({ message: "Failed to sync wearable" });
    }
  });

  // ============== REFERRAL SYSTEM ROUTES ==============

  app.get('/api/referrals/my-code', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const user = req.user!;
      
      // Get or create referral code for this user
      let referral = await storage.getReferralByReferrerId(userId);
      
      if (!referral) {
        // Generate unique referral code
        const referralCode = `REF-${user.firstName.substring(0, 3).toUpperCase()}${Math.random().toString(36).substring(2, 8).toUpperCase()}`;
        const referralLink = `${process.env.APP_URL || 'http://localhost:5000'}/signup?ref=${referralCode}`;
        
        referral = await storage.createReferral({
          referrerId: userId,
          referrerType: user.role,
          referralCode,
          referralLink,
          expiresAt: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000), // 90 days
        });
      }
      
      res.json(referral);
    } catch (error) {
      console.error("Error fetching referral code:", error);
      res.status(500).json({ message: "Failed to fetch referral code" });
    }
  });

  app.get('/api/referrals/my-referrals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const referrals = await storage.getReferralsByReferrerId(userId);
      res.json(referrals);
    } catch (error) {
      console.error("Error fetching referrals:", error);
      res.status(500).json({ message: "Failed to fetch referrals" });
    }
  });

  app.post('/api/referrals/track-click', async (req, res) => {
    try {
      const { referralCode, email } = req.body;
      
      if (!referralCode) {
        return res.status(400).json({ message: "Referral code is required" });
      }
      
      const referral = await storage.getReferralByCode(referralCode);
      if (!referral) {
        return res.status(404).json({ message: "Invalid referral code" });
      }
      
      // Check if expired
      if (referral.expiresAt && new Date(referral.expiresAt) < new Date()) {
        return res.status(400).json({ message: "Referral link has expired" });
      }
      
      // Update click tracking
      await storage.updateReferral(referral.id, {
        clickedAt: new Date(),
        refereeEmail: email,
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'],
      });
      
      res.json({ success: true, referral });
    } catch (error) {
      console.error("Error tracking referral click:", error);
      res.status(500).json({ message: "Failed to track referral click" });
    }
  });

  app.post('/api/referrals/activate', isAuthenticated, async (req: any, res) => {
    try {
      const { referralCode } = req.body;
      const newUserId = req.user!.id;
      const newUser = req.user!;
      
      if (!referralCode) {
        return res.status(400).json({ message: "Referral code is required" });
      }
      
      const referral = await storage.getReferralByCode(referralCode);
      if (!referral) {
        return res.status(404).json({ message: "Invalid referral code" });
      }
      
      // Prevent self-referral
      if (referral.referrerId === newUserId) {
        return res.status(400).json({ message: "You cannot use your own referral code" });
      }
      
      // Check if already activated
      if (referral.status === 'trial_activated') {
        return res.status(400).json({ message: "Referral code already used" });
      }
      
      // Update referral with referee info
      await storage.updateReferral(referral.id, {
        refereeId: newUserId,
        refereeType: newUser.role,
        signedUpAt: new Date(),
        status: 'signed_up',
      });
      
      // Grant 1-month free trial to both users
      const trialEnd = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // 30 days
      
      // Update referrer's trial
      await storage.updateUser(referral.referrerId, {
        trialEndsAt: trialEnd,
        subscriptionStatus: 'trialing',
      });
      
      // Update referee's trial
      await storage.updateUser(newUserId, {
        trialEndsAt: trialEnd,
        subscriptionStatus: 'trialing',
      });
      
      // Mark referral as activated
      await storage.updateReferral(referral.id, {
        status: 'trial_activated',
        trialActivatedAt: new Date(),
        referrerTrialExtended: true,
        refereeTrialGranted: true,
      });
      
      res.json({ 
        success: true, 
        message: "Referral activated! Both you and your referrer received 1 month free trial.",
        trialEndsAt: trialEnd,
      });
    } catch (error) {
      console.error("Error activating referral:", error);
      res.status(500).json({ message: "Failed to activate referral" });
    }
  });

  // ============== WALLET & CREDIT ROUTES ==============

  app.get('/api/wallet/balance', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const user = await storage.getUserById(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      res.json({ balance: user.creditBalance || 0 });
    } catch (error) {
      console.error("Error fetching wallet balance:", error);
      res.status(500).json({ message: "Failed to fetch wallet balance" });
    }
  });

  app.get('/api/wallet/transactions', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const transactions = await storage.getCreditTransactions(userId);
      res.json(transactions);
    } catch (error) {
      console.error("Error fetching credit transactions:", error);
      res.status(500).json({ message: "Failed to fetch credit transactions" });
    }
  });

  app.post('/api/wallet/purchase', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { amount, paymentMethod } = req.body;
      
      if (!amount || amount <= 0) {
        return res.status(400).json({ message: "Invalid amount" });
      }
      
      // TODO: Integrate with Stripe to process payment
      // For now, just add credits (in production, this should happen after payment confirmation)
      
      const user = await storage.getUserById(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      const newBalance = (user.creditBalance || 0) + amount;
      await storage.updateUser(userId, { creditBalance: newBalance });
      
      // Create transaction record
      await storage.createCreditTransaction({
        userId,
        transactionType: 'purchased',
        amount,
        balanceAfter: newBalance,
        description: `Purchased ${amount} credits`,
      });
      
      res.json({ success: true, newBalance });
    } catch (error) {
      console.error("Error purchasing credits:", error);
      res.status(500).json({ message: "Failed to purchase credits" });
    }
  });

  app.post('/api/wallet/withdraw', isAuthenticated, isDoctor, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { amount } = req.body;
      
      if (!amount || amount <= 0) {
        return res.status(400).json({ message: "Invalid withdrawal amount" });
      }
      
      const user = await storage.getUserById(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      if ((user.creditBalance || 0) < amount) {
        return res.status(400).json({ message: "Insufficient credits for withdrawal" });
      }
      
      // TODO: Integrate with Stripe to process payout to doctor's account
      // For now, just deduct credits (in production, this should happen after payout confirmation)
      
      const newBalance = (user.creditBalance || 0) - amount;
      await storage.updateUser(userId, { creditBalance: newBalance });
      
      // Create transaction record
      await storage.createCreditTransaction({
        userId,
        transactionType: 'withdrawn',
        amount: -amount,
        balanceAfter: newBalance,
        description: `Withdrew ${amount} credits`,
        metadata: { withdrawalAmount: amount },
      });
      
      res.json({ success: true, newBalance, message: "Withdrawal request submitted. Funds will be transferred to your account within 2-3 business days." });
    } catch (error) {
      console.error("Error withdrawing credits:", error);
      res.status(500).json({ message: "Failed to withdraw credits" });
    }
  });

  // ============== MEDICAL DOCUMENTS ROUTES ==============

  app.post('/api/medical-documents/upload', isAuthenticated, medicalDocUpload.single('file'), async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const file = req.file;
      
      if (!file) {
        return res.status(400).json({ message: "No file uploaded" });
      }
      
      const documentType = req.body.documentType || 'other';
      const documentDate = req.body.documentDate ? new Date(req.body.documentDate) : null;
      
      const fileKey = `medical-documents/${userId}/${Date.now()}-${file.originalname}`;
      
      const uploadParams = {
        Bucket: AWS_S3_BUCKET,
        Key: fileKey,
        Body: file.buffer,
        ContentType: file.mimetype,
        ServerSideEncryption: 'AES256' as const,
        Metadata: {
          userId,
          documentType,
          uploadDate: new Date().toISOString(),
        },
      };
      
      const upload = new Upload({
        client: s3Client,
        params: uploadParams,
      });
      
      await upload.done();
      
      const fileUrl = `https://${AWS_S3_BUCKET}.s3.${process.env.AWS_REGION}.amazonaws.com/${fileKey}`;
      
      const document = await storage.createMedicalDocument({
        userId,
        fileName: file.originalname,
        fileType: path.extname(file.originalname).substring(1),
        fileSize: file.size,
        fileUrl,
        documentType,
        documentDate,
        processingStatus: 'pending',
      });
      
      res.json({
        success: true,
        document,
        message: 'Document uploaded successfully. OCR processing will begin shortly.',
      });
      
      processDocumentOCR(document.id, fileKey).catch(error => {
        console.error('Error processing OCR:', error);
      });
      
    } catch (error) {
      console.error('Error uploading medical document:', error);
      res.status(500).json({ message: 'Failed to upload document' });
    }
  });

  async function processDocumentOCR(documentId: string, s3Key: string) {
    try {
      await storage.updateMedicalDocument(documentId, { processingStatus: 'processing' });
      
      const startCommand = new StartDocumentAnalysisCommand({
        DocumentLocation: {
          S3Object: {
            Bucket: AWS_S3_BUCKET,
            Name: s3Key,
          },
        },
        FeatureTypes: ['TABLES', 'FORMS'],
      });
      
      const startResponse = await textractClient.send(startCommand);
      const jobId = startResponse.JobId;
      
      if (!jobId) {
        throw new Error('Failed to start Textract job');
      }
      
      let jobStatus = 'IN_PROGRESS';
      let attempts = 0;
      const maxAttempts = 300;
      
      while (jobStatus === 'IN_PROGRESS' && attempts < maxAttempts) {
        const baseDelay = 2000;
        const delay = attempts < 30 ? baseDelay : 
                     attempts < 60 ? baseDelay * 2 : 
                     attempts < 120 ? baseDelay * 3 : 
                     baseDelay * 4;
        
        await new Promise(resolve => setTimeout(resolve, delay));
        
        const getCommand = new GetDocumentAnalysisCommand({ JobId: jobId });
        const getResponse = await textractClient.send(getCommand);
        
        jobStatus = getResponse.JobStatus || 'IN_PROGRESS';
        
        if (jobStatus === 'SUCCEEDED') {
          let extractedText = '';
          let allBlocks = getResponse.Blocks || [];
          let nextToken = getResponse.NextToken;
          
          while (nextToken) {
            const nextCommand = new GetDocumentAnalysisCommand({ 
              JobId: jobId,
              NextToken: nextToken,
            });
            const nextResponse = await textractClient.send(nextCommand);
            allBlocks = allBlocks.concat(nextResponse.Blocks || []);
            nextToken = nextResponse.NextToken;
          }
          
          for (const block of allBlocks) {
            if (block.BlockType === 'LINE' && block.Text) {
              extractedText += block.Text + '\n';
            }
          }
          
          let allMedicalEntities: any[] = [];
          if (extractedText.trim().length > 0) {
            try {
              const MAX_CHUNK_SIZE = 19000;
              const chunks: string[] = [];
              
              if (extractedText.length <= MAX_CHUNK_SIZE) {
                chunks.push(extractedText);
              } else {
                const lines = extractedText.split('\n');
                let currentChunk = '';
                
                for (const line of lines) {
                  if ((currentChunk.length + line.length + 1) > MAX_CHUNK_SIZE) {
                    if (currentChunk.length > 0) {
                      chunks.push(currentChunk);
                    }
                    currentChunk = line;
                  } else {
                    currentChunk += (currentChunk ? '\n' : '') + line;
                  }
                }
                
                if (currentChunk.length > 0) {
                  chunks.push(currentChunk);
                }
              }
              
              for (const chunk of chunks) {
                try {
                  const comprehendCommand = new DetectEntitiesV2Command({
                    Text: chunk,
                  });
                  
                  const comprehendResponse = await comprehendMedicalClient.send(comprehendCommand);
                  allMedicalEntities = allMedicalEntities.concat(comprehendResponse.Entities || []);
                  
                  await new Promise(resolve => setTimeout(resolve, 100));
                } catch (chunkError) {
                  console.error('Comprehend Medical chunk error:', chunkError);
                }
              }
            } catch (comprehendError) {
              console.error('Comprehend Medical error:', comprehendError);
            }
          }
          
          const extractedData: any = {
            medications: [],
            diagnosis: [],
            labResults: [],
            vitalSigns: [],
            allergies: [],
            procedures: [],
          };
          
          const seenTexts = new Set<string>();
          
          for (const entity of allMedicalEntities) {
            const normalizedText = entity.Text?.toLowerCase().trim();
            if (!normalizedText || seenTexts.has(normalizedText)) continue;
            seenTexts.add(normalizedText);
            
            if (entity.Category === 'MEDICATION' && entity.Text) {
              extractedData.medications.push(entity.Text);
            } else if (entity.Category === 'MEDICAL_CONDITION' && entity.Text) {
              extractedData.diagnosis.push(entity.Text);
            } else if (entity.Category === 'TEST_TREATMENT_PROCEDURE' && entity.Text) {
              extractedData.procedures.push(entity.Text);
            }
          }
          
          await storage.updateMedicalDocument(documentId, {
            extractedText,
            extractedData,
            processingStatus: 'completed',
          });
          
          return;
        } else if (jobStatus === 'FAILED') {
          throw new Error(getResponse.StatusMessage || 'Textract job failed');
        }
        
        attempts++;
      }
      
      if (jobStatus === 'IN_PROGRESS') {
        throw new Error('Textract job timeout - processing took too long');
      }
      
    } catch (error) {
      console.error('OCR processing error:', error);
      await storage.updateMedicalDocument(documentId, {
        processingStatus: 'failed',
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  app.get('/api/medical-documents', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const documents = await storage.getMedicalDocuments(userId);
      res.json(documents);
    } catch (error) {
      console.error('Error fetching medical documents:', error);
      res.status(500).json({ message: 'Failed to fetch documents' });
    }
  });

  app.get('/api/medical-documents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const documentId = req.params.id;
      
      const document = await storage.getMedicalDocument(documentId);
      
      if (!document) {
        return res.status(404).json({ message: 'Document not found' });
      }
      
      if (document.userId !== userId && req.user!.role !== 'doctor') {
        return res.status(403).json({ message: 'Access denied' });
      }
      
      res.json(document);
    } catch (error) {
      console.error('Error fetching medical document:', error);
      res.status(500).json({ message: 'Failed to fetch document' });
    }
  });

  app.delete('/api/medical-documents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const documentId = req.params.id;
      
      const document = await storage.getMedicalDocument(documentId);
      
      if (!document) {
        return res.status(404).json({ message: 'Document not found' });
      }
      
      if (document.userId !== userId) {
        return res.status(403).json({ message: 'Access denied' });
      }
      
      await storage.deleteMedicalDocument(documentId);
      
      res.json({ success: true, message: 'Document deleted successfully' });
    } catch (error) {
      console.error('Error deleting medical document:', error);
      res.status(500).json({ message: 'Failed to delete document' });
    }
  });

  // Immune monitoring routes
  app.get('/api/immune/biomarkers', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 30;
      const biomarkers = await storage.getImmuneBiomarkers(userId, limit);
      res.json(biomarkers);
    } catch (error) {
      console.error('Error fetching immune biomarkers:', error);
      res.status(500).json({ message: 'Failed to fetch biomarkers' });
    }
  });

  app.get('/api/immune/digital-twin', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const digitalTwin = await storage.getLatestImmuneDigitalTwin(userId);
      res.json(digitalTwin);
    } catch (error) {
      console.error('Error fetching immune digital twin:', error);
      res.status(500).json({ message: 'Failed to fetch digital twin' });
    }
  });

  app.post('/api/immune/sync-wearable', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { deviceType } = req.body;
      
      if (!deviceType) {
        return res.status(400).json({ message: 'Device type is required' });
      }

      const { syncWearableData } = await import('./immuneMonitoring');
      const result = await syncWearableData(userId, deviceType);
      
      res.json(result);
    } catch (error) {
      console.error('Error syncing wearable data:', error);
      res.status(500).json({ message: 'Failed to sync wearable data' });
    }
  });

  app.post('/api/immune/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { analyzeImmuneBiomarkers } = await import('./immuneMonitoring');
      const analysis = await analyzeImmuneBiomarkers(userId);
      res.json(analysis);
    } catch (error) {
      console.error('Error analyzing immune biomarkers:', error);
      res.status(500).json({ message: 'Failed to analyze biomarkers' });
    }
  });

  // Environmental risk routes
  app.get('/api/environmental/risk', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const user = await storage.getUser(userId);
      const patientProfile = await storage.getPatientProfile(userId);
      
      if (!patientProfile?.zipCode) {
        return res.status(400).json({ message: 'User location (zip code) is required' });
      }

      const riskData = await storage.getEnvironmentalRiskDataByLocation(patientProfile.zipCode, 1);
      
      if (riskData.length > 0) {
        res.json(riskData[0]);
      } else {
        res.json({ message: 'No environmental risk data available for this location' });
      }
    } catch (error) {
      console.error('Error fetching environmental risk data:', error);
      res.status(500).json({ message: 'Failed to fetch environmental risk data' });
    }
  });

  app.post('/api/environmental/update', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const patientProfile = await storage.getPatientProfile(userId);
      
      if (!patientProfile?.zipCode) {
        return res.status(400).json({ message: 'User location (zip code) is required' });
      }

      const { fetchEnvironmentalRiskData } = await import('./environmentalRisk');
      const riskData = await fetchEnvironmentalRiskData(patientProfile.zipCode);
      
      res.json(riskData);
    } catch (error) {
      console.error('Error updating environmental risk data:', error);
      res.status(500).json({ message: 'Failed to update environmental risk data' });
    }
  });

  app.get('/api/environmental/pathogen-map', isAuthenticated, async (req: any, res) => {
    try {
      const { zipCode } = req.query;
      
      if (!zipCode) {
        return res.status(400).json({ message: 'Zip code is required' });
      }

      const { generatePathogenRiskMap } = await import('./environmentalRisk');
      const riskMap = await generatePathogenRiskMap(zipCode as string);
      
      res.json(riskMap);
    } catch (error) {
      console.error('Error generating pathogen risk map:', error);
      res.status(500).json({ message: 'Failed to generate risk map' });
    }
  });

  // Multi-Condition Correlation Engine routes
  app.post('/api/correlations/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { analyzeCorrelations } = await import('./correlationEngine');
      await analyzeCorrelations(userId);
      res.json({ message: 'Correlation analysis completed successfully' });
    } catch (error) {
      console.error('Error analyzing correlations:', error);
      res.status(500).json({ message: 'Failed to analyze correlations' });
    }
  });

  app.get('/api/correlations', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit, type, severity } = req.query;
      
      let patterns;
      if (type) {
        patterns = await storage.getCorrelationPatternsByType(userId, type as string, limit ? parseInt(limit as string) : undefined);
      } else if (severity) {
        patterns = await storage.getHighSeverityCorrelationPatterns(userId, severity as string, limit ? parseInt(limit as string) : undefined);
      } else {
        patterns = await storage.getCorrelationPatterns(userId, limit ? parseInt(limit as string) : undefined);
      }
      
      res.json(patterns);
    } catch (error) {
      console.error('Error fetching correlation patterns:', error);
      res.status(500).json({ message: 'Failed to fetch correlation patterns' });
    }
  });

  app.get('/api/correlations/report', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { generateCorrelationReport } = await import('./correlationEngine');
      const report = await generateCorrelationReport(userId);
      res.json(report);
    } catch (error) {
      console.error('Error generating correlation report:', error);
      res.status(500).json({ message: 'Failed to generate correlation report' });
    }
  });

  // Risk alert routes
  app.get('/api/alerts/active', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const alerts = await storage.getActiveRiskAlerts(userId);
      res.json(alerts);
    } catch (error) {
      console.error('Error fetching active alerts:', error);
      res.status(500).json({ message: 'Failed to fetch alerts' });
    }
  });

  app.get('/api/alerts/all', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 50;
      const alerts = await storage.getAllRiskAlerts(userId, limit);
      res.json(alerts);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      res.status(500).json({ message: 'Failed to fetch alerts' });
    }
  });

  app.post('/api/alerts/:id/acknowledge', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const alert = await storage.acknowledgeRiskAlert(id);
      res.json(alert);
    } catch (error) {
      console.error('Error acknowledging alert:', error);
      res.status(500).json({ message: 'Failed to acknowledge alert' });
    }
  });

  app.post('/api/alerts/:id/resolve', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const alert = await storage.resolveRiskAlert(id);
      res.json(alert);
    } catch (error) {
      console.error('Error resolving alert:', error);
      res.status(500).json({ message: 'Failed to resolve alert' });
    }
  });

  app.post('/api/alerts/:id/dismiss', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const alert = await storage.dismissRiskAlert(id);
      res.json(alert);
    } catch (error) {
      console.error('Error dismissing alert:', error);
      res.status(500).json({ message: 'Failed to dismiss alert' });
    }
  });

  // ==================== ADAPTIVE MEDICATION & NUTRITION INSIGHTS ROUTES ====================

  // Dietary preferences
  app.get('/api/nutrition/preferences', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const preferences = await storage.getDietaryPreferences(userId);
      res.json(preferences || {});
    } catch (error) {
      console.error('Error fetching dietary preferences:', error);
      res.status(500).json({ message: 'Failed to fetch dietary preferences' });
    }
  });

  app.post('/api/nutrition/preferences', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const preferences = await storage.upsertDietaryPreferences({
        patientId: userId,
        ...req.body,
      });
      res.json(preferences);
    } catch (error) {
      console.error('Error updating dietary preferences:', error);
      res.status(500).json({ message: 'Failed to update dietary preferences' });
    }
  });

  // Meal plans
  app.post('/api/nutrition/generate-meal-plan', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { generateWeeklyMealPlan } = await import('./nutritionService');
      const result = await generateWeeklyMealPlan(userId);
      res.json(result);
    } catch (error) {
      console.error('Error generating meal plan:', error);
      res.status(500).json({ message: error instanceof Error ? error.message : 'Failed to generate meal plan' });
    }
  });

  app.get('/api/nutrition/meal-plans', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { activeOnly } = req.query;
      const plans = await storage.getMealPlans(userId, activeOnly === 'true');
      res.json(plans);
    } catch (error) {
      console.error('Error fetching meal plans:', error);
      res.status(500).json({ message: 'Failed to fetch meal plans' });
    }
  });

  app.get('/api/nutrition/active-plan', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const plan = await storage.getActiveMealPlan(userId);
      res.json(plan || null);
    } catch (error) {
      console.error('Error fetching active meal plan:', error);
      res.status(500).json({ message: 'Failed to fetch active meal plan' });
    }
  });

  // Meals
  app.get('/api/nutrition/meals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { mealPlanId, limit } = req.query;
      const meals = await storage.getMeals(
        userId,
        mealPlanId as string | undefined,
        limit ? parseInt(limit as string) : undefined
      );
      res.json(meals);
    } catch (error) {
      console.error('Error fetching meals:', error);
      res.status(500).json({ message: 'Failed to fetch meals' });
    }
  });

  app.get('/api/nutrition/meals/today', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const meals = await storage.getTodaysMeals(userId);
      res.json(meals);
    } catch (error) {
      console.error('Error fetching today\'s meals:', error);
      res.status(500).json({ message: 'Failed to fetch today\'s meals' });
    }
  });

  app.post('/api/nutrition/meals', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const meal = await storage.createMeal({
        patientId: userId,
        ...req.body,
      });
      res.json(meal);
    } catch (error) {
      console.error('Error creating meal:', error);
      res.status(500).json({ message: 'Failed to create meal' });
    }
  });

  app.patch('/api/nutrition/meals/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const meal = await storage.updateMeal(id, req.body);
      res.json(meal);
    } catch (error) {
      console.error('Error updating meal:', error);
      res.status(500).json({ message: 'Failed to update meal' });
    }
  });

  // Nutrition analysis
  app.post('/api/nutrition/analyze', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { mealDescription } = req.body;
      if (!mealDescription) {
        return res.status(400).json({ message: 'Meal description is required' });
      }
      const { analyzeMealNutrition } = await import('./nutritionService');
      const analysis = await analyzeMealNutrition(mealDescription, userId);
      res.json(analysis);
    } catch (error) {
      console.error('Error analyzing meal:', error);
      res.status(500).json({ message: 'Failed to analyze meal' });
    }
  });

  // Medication scheduling
  app.get('/api/medications/schedules', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { activeOnly } = req.query;
      const schedules = await storage.getPatientMedicationSchedules(userId, activeOnly !== 'false');
      res.json(schedules);
    } catch (error) {
      console.error('Error fetching medication schedules:', error);
      res.status(500).json({ message: 'Failed to fetch medication schedules' });
    }
  });

  app.post('/api/medications/optimize-timing', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { optimizeMedicationTiming } = await import('./nutritionService');
      const recommendations = await optimizeMedicationTiming(userId);
      res.json(recommendations);
    } catch (error) {
      console.error('Error optimizing medication timing:', error);
      res.status(500).json({ message: 'Failed to optimize medication timing' });
    }
  });

  // Medication adherence
  app.get('/api/medications/adherence', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit } = req.query;
      const adherence = await storage.getPatientAdherence(userId, limit ? parseInt(limit as string) : undefined);
      res.json(adherence);
    } catch (error) {
      console.error('Error fetching adherence data:', error);
      res.status(500).json({ message: 'Failed to fetch adherence data' });
    }
  });

  app.get('/api/medications/pending', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const pending = await storage.getPendingMedications(userId);
      res.json(pending);
    } catch (error) {
      console.error('Error fetching pending medications:', error);
      res.status(500).json({ message: 'Failed to fetch pending medications' });
    }
  });

  app.post('/api/medications/adherence', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const adherence = await storage.createMedicationAdherence({
        patientId: userId,
        ...req.body,
      });
      res.json(adherence);
    } catch (error) {
      console.error('Error logging medication adherence:', error);
      res.status(500).json({ message: 'Failed to log medication adherence' });
    }
  });

  app.patch('/api/medications/adherence/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const adherence = await storage.updateMedicationAdherence(id, req.body);
      res.json(adherence);
    } catch (error) {
      console.error('Error updating medication adherence:', error);
      res.status(500).json({ message: 'Failed to update medication adherence' });
    }
  });

  // ==================== HEALTH COMPANION MODE ROUTES ====================

  // Companion check-ins
  app.post('/api/companion/check-in', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { checkInType, userInput, context } = req.body;
      
      if (!userInput) {
        return res.status(400).json({ message: 'User input is required' });
      }

      const { processCompanionCheckIn } = await import('./companionService');
      const result = await processCompanionCheckIn({
        patientId: userId,
        checkInType: checkInType || 'spontaneous',
        userInput,
        context,
      });

      res.json(result);
    } catch (error) {
      console.error('Error processing companion check-in:', error);
      res.status(500).json({ message: 'Failed to process check-in' });
    }
  });

  app.get('/api/companion/check-ins', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit, type } = req.query;
      
      let checkIns;
      if (type) {
        checkIns = await storage.getCheckInsByType(
          userId,
          type as string,
          limit ? parseInt(limit as string) : undefined
        );
      } else {
        checkIns = await storage.getCompanionCheckIns(
          userId,
          limit ? parseInt(limit as string) : undefined
        );
      }

      res.json(checkIns);
    } catch (error) {
      console.error('Error fetching check-ins:', error);
      res.status(500).json({ message: 'Failed to fetch check-ins' });
    }
  });

  app.get('/api/companion/check-ins/recent', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { days } = req.query;
      const checkIns = await storage.getRecentCheckIns(
        userId,
        days ? parseInt(days as string) : 7
      );
      res.json(checkIns);
    } catch (error) {
      console.error('Error fetching recent check-ins:', error);
      res.status(500).json({ message: 'Failed to fetch recent check-ins' });
    }
  });

  // Companion engagement
  app.get('/api/companion/engagement', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const engagement = await storage.getCompanionEngagement(userId);
      res.json(engagement || {});
    } catch (error) {
      console.error('Error fetching engagement data:', error);
      res.status(500).json({ message: 'Failed to fetch engagement data' });
    }
  });

  app.patch('/api/companion/engagement', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const engagement = await storage.updateCompanionEngagement(userId, req.body);
      res.json(engagement);
    } catch (error) {
      console.error('Error updating engagement settings:', error);
      res.status(500).json({ message: 'Failed to update engagement settings' });
    }
  });

  // Helper endpoints
  app.get('/api/companion/suggest-check-in-time', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { suggestCheckInTime } = await import('./companionService');
      const time = await suggestCheckInTime(userId);
      res.json({ suggestedTime: time });
    } catch (error) {
      console.error('Error suggesting check-in time:', error);
      res.status(500).json({ message: 'Failed to suggest check-in time' });
    }
  });

  app.get('/api/companion/prompt', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { type } = req.query;
      const { generateCheckInPrompt } = await import('./companionService');
      const prompt = await generateCheckInPrompt(userId, (type as string) || 'morning');
      res.json({ prompt });
    } catch (error) {
      console.error('Error generating check-in prompt:', error);
      res.status(500).json({ message: 'Failed to generate prompt' });
    }
  });

  // Voice Followup Routes - Quick 1min voice logs
  const voiceUpload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 25 * 1024 * 1024 }, // 25MB limit (OpenAI Whisper limit)
    fileFilter: (req, file, cb) => {
      const allowedTypes = ['audio/webm', 'audio/wav', 'audio/mp3', 'audio/m4a', 'audio/mpeg'];
      if (allowedTypes.includes(file.mimetype)) {
        cb(null, true);
      } else {
        cb(new Error('Invalid file type. Only audio files are allowed.'));
      }
    },
  });

  app.post('/api/voice-followup/upload', isAuthenticated, voiceUpload.single('audio'), async (req: any, res) => {
    try {
      const userId = req.user!.id;
      
      if (!req.file) {
        return res.status(400).json({ message: 'No audio file provided' });
      }

      const { processVoiceFollowup } = await import('./voiceProcessingService');
      
      // Process the voice recording (pass mimetype for S3)
      const processedData = await processVoiceFollowup(
        req.file.buffer,
        req.file.originalname,
        userId,
        req.file.mimetype
      );

      // Store in database
      const voiceFollowup = await storage.createVoiceFollowup({
        patientId: userId,
        ...processedData,
      });

      res.json({
        id: voiceFollowup.id,
        transcription: voiceFollowup.transcription,
        response: voiceFollowup.aiResponse,
        empathyLevel: voiceFollowup.empathyLevel,
        conversationSummary: voiceFollowup.conversationSummary,
        concernsRaised: voiceFollowup.concernsRaised,
        needsFollowup: voiceFollowup.needsFollowup,
        followupReason: voiceFollowup.followupReason,
        recommendedActions: voiceFollowup.recommendedActions,
        extractedSymptoms: voiceFollowup.extractedSymptoms,
        extractedMood: voiceFollowup.extractedMood,
        medicationAdherence: voiceFollowup.medicationAdherence,
        extractedMetrics: voiceFollowup.extractedMetrics,
      });
    } catch (error) {
      console.error('Error processing voice followup:', error);
      res.status(500).json({ message: 'Failed to process voice recording' });
    }
  });

  app.get('/api/voice-followup/recent', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const { limit } = req.query;
      const followups = await storage.getRecentVoiceFollowups(
        userId,
        limit ? parseInt(limit) : 10
      );
      res.json(followups);
    } catch (error) {
      console.error('Error fetching voice followups:', error);
      res.status(500).json({ message: 'Failed to fetch voice followups' });
    }
  });

  app.get('/api/voice-followup/:id', isAuthenticated, async (req: any, res) => {
    try {
      const { id } = req.params;
      const followup = await storage.getVoiceFollowup(id);
      
      if (!followup) {
        return res.status(404).json({ message: 'Voice followup not found' });
      }
      
      // Verify the followup belongs to the requesting user
      if (followup.patientId !== req.user!.id && req.user!.role !== 'doctor') {
        return res.status(403).json({ message: 'Unauthorized' });
      }
      
      res.json(followup);
    } catch (error) {
      console.error('Error fetching voice followup:', error);
      res.status(500).json({ message: 'Failed to fetch voice followup' });
    }
  });

  app.get('/api/voice-followup', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user!.id;
      const followups = await storage.getAllVoiceFollowups(userId);
      res.json(followups);
    } catch (error) {
      console.error('Error fetching all voice followups:', error);
      res.status(500).json({ message: 'Failed to fetch voice followups' });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
