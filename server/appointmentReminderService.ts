import { SESClient, SendEmailCommand } from '@aws-sdk/client-ses';
import twilio from 'twilio';
import type { Storage } from './storage';
import { addDays, isBefore, isAfter, startOfDay, endOfDay } from 'date-fns';

const sesClient = new SESClient({ region: process.env.AWS_REGION || 'us-east-1' });

interface ReminderConfig {
  enableSMS: boolean;
  enableEmail: boolean;
  daysBefore: number[];
  hoursBefore: number[];
}

const DEFAULT_REMINDER_CONFIG: ReminderConfig = {
  enableSMS: true,
  enableEmail: true,
  daysBefore: [1, 7],
  hoursBefore: [24],
};

class AppointmentReminderService {
  private storage: Storage;
  private twilioClient: twilio.Twilio;

  constructor(storage: Storage) {
    this.storage = storage;
    this.twilioClient = twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );
  }

  async sendDailyReminders(): Promise<{
    sent: number;
    failed: number;
    errors: string[];
  }> {
    const results = {
      sent: 0,
      failed: 0,
      errors: [] as string[],
    };

    try {
      const upcomingAppointments = await this.getUpcomingAppointments();

      for (const appointment of upcomingAppointments) {
        try {
          const patient = await this.storage.getUser(appointment.patientId);
          const doctor = await this.storage.getUser(appointment.doctorId);

          if (!patient || !doctor) {
            results.failed++;
            results.errors.push(`Missing patient or doctor for appointment ${appointment.id}`);
            continue;
          }

          const reminderConfig = DEFAULT_REMINDER_CONFIG;

          if (reminderConfig.enableSMS && patient.phoneNumber) {
            await this.sendSMSReminder(appointment, patient, doctor);
          }

          if (reminderConfig.enableEmail && patient.email) {
            await this.sendEmailReminder(appointment, patient, doctor);
          }

          const newReminder = {
            type: 'daily_reminder',
            sentAt: new Date().toISOString(),
          };
          const remindersSent = appointment.remindersSent || [];
          
          await this.storage.updateAppointment(appointment.id, {
            ...appointment,
            remindersSent: [...remindersSent, newReminder] as any,
          });

          results.sent++;
        } catch (error) {
          results.failed++;
          results.errors.push(
            `Failed to send reminder for appointment ${appointment.id}: ${
              error instanceof Error ? error.message : 'Unknown error'
            }`
          );
        }
      }
    } catch (error) {
      results.errors.push(
        `Failed to fetch upcoming appointments: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }

    return results;
  }

  private async getUpcomingAppointments() {
    const tomorrow = startOfDay(addDays(new Date(), 1));
    const dayAfterTomorrow = endOfDay(addDays(new Date(), 1));

    const allDoctors = await this.storage.getAllDoctors();
    const allAppointments: any[] = [];

    for (const doctor of allDoctors) {
      const doctorAppointments = await this.storage.getAppointmentsByDoctor(doctor.id);
      allAppointments.push(...doctorAppointments);
    }
    
    return allAppointments.filter((appointment) => {
      const appointmentDate = new Date(appointment.startTime);
      const reminders = appointment.remindersSent || [];
      const hasRecentReminder = reminders.some((r: any) => {
        const sentAt = new Date(r.sentAt);
        const hoursSince = (Date.now() - sentAt.getTime()) / (1000 * 60 * 60);
        return hoursSince < 12;
      });
      
      return (
        !hasRecentReminder &&
        (appointment.status === 'confirmed' || appointment.status === 'scheduled') &&
        isAfter(appointmentDate, tomorrow) &&
        isBefore(appointmentDate, dayAfterTomorrow)
      );
    });
  }

  private async sendSMSReminder(appointment: any, patient: any, doctor: any): Promise<void> {
    const appointmentDate = new Date(appointment.startTime);
    const formattedDate = appointmentDate.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
    const formattedTime = appointmentDate.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
    });

    const message = `Reminder: You have an appointment with Dr. ${doctor.firstName} ${doctor.lastName} on ${formattedDate} at ${formattedTime}. ${appointment.location || 'Location TBD'}. Reply CONFIRM to confirm or CANCEL to cancel.`;

    await this.twilioClient.messages.create({
      body: message,
      from: process.env.TWILIO_PHONE_NUMBER,
      to: patient.phoneNumber,
    });
  }

  private async sendEmailReminder(appointment: any, patient: any, doctor: any): Promise<void> {
    const appointmentDate = new Date(appointment.startTime);
    const formattedDate = appointmentDate.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
    const formattedTime = appointmentDate.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
    });

    const emailHtml = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: #4f46e5; color: white; padding: 20px; text-align: center; }
          .content { background-color: #f9fafb; padding: 30px; }
          .appointment-details { background-color: white; padding: 20px; margin: 20px 0; border-left: 4px solid #4f46e5; }
          .button { display: inline-block; padding: 12px 30px; background-color: #4f46e5; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }
          .footer { text-align: center; padding: 20px; color: #6b7280; font-size: 12px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Appointment Reminder</h1>
          </div>
          <div class="content">
            <p>Hello ${patient.firstName},</p>
            <p>This is a reminder about your upcoming appointment:</p>
            <div class="appointment-details">
              <p><strong>Doctor:</strong> Dr. ${doctor.firstName} ${doctor.lastName}</p>
              <p><strong>Date:</strong> ${formattedDate}</p>
              <p><strong>Time:</strong> ${formattedTime}</p>
              <p><strong>Location:</strong> ${appointment.location || 'To be determined'}</p>
              ${appointment.notes ? `<p><strong>Notes:</strong> ${appointment.notes}</p>` : ''}
            </div>
            <p>Please arrive 10 minutes early to complete any necessary paperwork.</p>
            <div style="text-align: center;">
              <a href="#" class="button">Confirm Appointment</a>
              <a href="#" class="button" style="background-color: #dc2626;">Cancel Appointment</a>
            </div>
          </div>
          <div class="footer">
            <p>This is an automated reminder from Followup AI</p>
            <p>Please do not reply to this email</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const command = new SendEmailCommand({
      Source: process.env.SES_FROM_EMAIL || 'noreply@followupai.com',
      Destination: {
        ToAddresses: [patient.email],
      },
      Message: {
        Subject: {
          Data: `Appointment Reminder - ${formattedDate}`,
        },
        Body: {
          Html: {
            Data: emailHtml,
          },
          Text: {
            Data: `Reminder: You have an appointment with Dr. ${doctor.firstName} ${doctor.lastName} on ${formattedDate} at ${formattedTime}. Location: ${appointment.location || 'TBD'}`,
          },
        },
      },
    });

    await sesClient.send(command);
  }

  async sendImmediateReminder(appointmentId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    try {
      const appointment = await this.storage.getAppointment(appointmentId);
      if (!appointment) {
        return { success: false, message: 'Appointment not found' };
      }

      const patient = await this.storage.getUser(appointment.patientId);
      const doctor = await this.storage.getUser(appointment.doctorId);

      if (!patient || !doctor) {
        return { success: false, message: 'Patient or doctor not found' };
      }

      if (patient.phoneNumber) {
        await this.sendSMSReminder(appointment, patient, doctor);
      }

      if (patient.email) {
        await this.sendEmailReminder(appointment, patient, doctor);
      }

      return { success: true, message: 'Reminder sent successfully' };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async processReminderResponse(from: string, message: string): Promise<void> {
    const normalizedMessage = message.trim().toUpperCase();

    if (normalizedMessage === 'CONFIRM') {
      const patient = await this.storage.getUserByPhoneNumber(from);
      if (patient) {
        const appointments = await this.storage.getAppointmentsByPatient(patient.id);
        const upcoming = appointments.filter((apt) => 
          new Date(apt.startTime) > new Date() && apt.status === 'confirmed'
        );
        
        if (upcoming.length > 0) {
          await this.storage.updateAppointment(upcoming[0].id, {
            ...upcoming[0],
            status: 'confirmed',
            confirmedAt: new Date(),
          });
        }
      }
    } else if (normalizedMessage === 'CANCEL') {
      const patient = await this.storage.getUserByPhoneNumber(from);
      if (patient) {
        const appointments = await this.storage.getAppointmentsByPatient(patient.id);
        const upcoming = appointments.filter((apt) => 
          new Date(apt.startTime) > new Date() && apt.status === 'confirmed'
        );
        
        if (upcoming.length > 0) {
          await this.storage.updateAppointment(upcoming[0].id, {
            ...upcoming[0],
            status: 'cancelled',
            cancelledAt: new Date(),
          });
        }
      }
    }
  }
}

export let appointmentReminderService: AppointmentReminderService;

export function initAppointmentReminderService(storage: Storage) {
  appointmentReminderService = new AppointmentReminderService(storage);
}
