import { google } from 'googleapis';
import PDFDocument from 'pdfkit';
import { Readable } from 'stream';

let connectionSettings: any;

async function getAccessToken() {
  if (connectionSettings && connectionSettings.settings.expires_at && new Date(connectionSettings.settings.expires_at).getTime() > Date.now()) {
    return connectionSettings.settings.access_token;
  }
  
  const hostname = process.env.REPLIT_CONNECTORS_HOSTNAME;
  const xReplitToken = process.env.REPL_IDENTITY 
    ? 'repl ' + process.env.REPL_IDENTITY 
    : process.env.WEB_REPL_RENEWAL 
    ? 'depl ' + process.env.WEB_REPL_RENEWAL 
    : null;

  if (!xReplitToken) {
    throw new Error('X_REPLIT_TOKEN not found for repl/depl');
  }

  connectionSettings = await fetch(
    'https://' + hostname + '/api/v2/connection?include_secrets=true&connector_names=google-drive',
    {
      headers: {
        'Accept': 'application/json',
        'X_REPLIT_TOKEN': xReplitToken
      }
    }
  ).then(res => res.json()).then(data => data.items?.[0]);

  const accessToken = connectionSettings?.settings?.access_token || connectionSettings.settings?.oauth?.credentials?.access_token;

  if (!connectionSettings || !accessToken) {
    throw new Error('Google Drive not connected');
  }
  return accessToken;
}

async function getUncachableGoogleDriveClient() {
  const accessToken = await getAccessToken();

  const oauth2Client = new google.auth.OAuth2();
  oauth2Client.setCredentials({
    access_token: accessToken
  });

  return google.drive({ version: 'v3', auth: oauth2Client });
}

interface DoctorApplicationData {
  email: string;
  firstName: string;
  lastName: string;
  organization: string;
  medicalLicenseNumber: string;
  licenseCountry: string;
  submittedAt: Date;
}

export async function generateDoctorApplicationPDF(data: DoctorApplicationData): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({
      size: 'A4',
      margins: { top: 50, bottom: 50, left: 50, right: 50 }
    });
    
    const chunks: Buffer[] = [];
    
    doc.on('data', (chunk) => chunks.push(chunk));
    doc.on('end', () => resolve(Buffer.concat(chunks)));
    doc.on('error', reject);

    // Header
    doc.fontSize(24).fillColor('#7C3AED').text('Followup AI', { align: 'center' });
    doc.fontSize(18).fillColor('#000000').text('Doctor Account Application', { align: 'center' });
    doc.moveDown(2);

    // Application Details
    doc.fontSize(14).fillColor('#7C3AED').text('Application Information', { underline: true });
    doc.moveDown(0.5);

    doc.fontSize(12).fillColor('#000000');
    doc.text(`Name: ${data.firstName} ${data.lastName}`);
    doc.moveDown(0.5);
    doc.text(`Email: ${data.email}`);
    doc.moveDown(0.5);
    doc.text(`Organization: ${data.organization}`);
    doc.moveDown(0.5);
    doc.text(`Medical License Number: ${data.medicalLicenseNumber}`);
    doc.moveDown(0.5);
    doc.text(`License Country: ${data.licenseCountry}`);
    doc.moveDown(0.5);
    doc.text(`Submitted: ${data.submittedAt.toLocaleString()}`);
    doc.moveDown(2);

    // Status
    doc.fontSize(14).fillColor('#7C3AED').text('Verification Status', { underline: true });
    doc.moveDown(0.5);
    doc.fontSize(12).fillColor('#000000');
    doc.text('Status: Pending Admin Verification');
    doc.moveDown(0.5);
    doc.text('Email Verified: Pending');
    doc.moveDown(0.5);
    doc.text('Phone Verified: Pending');
    doc.moveDown(0.5);
    doc.text('License Verified: Pending');
    doc.moveDown(2);

    // Footer
    doc.fontSize(10).fillColor('#666666');
    doc.text('This application requires admin approval before account activation.', { align: 'center' });
    doc.text('Followup AI - HIPAA-Compliant Health Platform', { align: 'center' });

    doc.end();
  });
}

export async function uploadDoctorApplicationToGoogleDrive(
  doctorData: DoctorApplicationData,
  pdfBuffer: Buffer
): Promise<string> {
  try {
    const drive = await getUncachableGoogleDriveClient();
    
    const fileName = `Doctor_Application_${doctorData.lastName}_${doctorData.firstName}_${Date.now()}.pdf`;
    
    const fileMetadata = {
      name: fileName,
      mimeType: 'application/pdf',
      parents: [] // Root folder, you can specify a folder ID if needed
    };

    const media = {
      mimeType: 'application/pdf',
      body: Readable.from(pdfBuffer)
    };

    const response = await drive.files.create({
      requestBody: fileMetadata,
      media: media,
      fields: 'id, name, webViewLink'
    });

    console.log(`[GOOGLE DRIVE] Doctor application uploaded: ${response.data.name} (${response.data.id})`);
    return response.data.webViewLink || response.data.id || '';
  } catch (error: any) {
    console.error('[GOOGLE DRIVE] Error uploading doctor application:', error.message);
    throw error;
  }
}
