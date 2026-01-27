import { NextRequest, NextResponse } from "next/server";

const REALM = "Followup AI";

function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

function parseBasicAuth(
  authHeader: string | null
): { username: string; password: string } | null {
  if (!authHeader || !authHeader.startsWith("Basic ")) {
    return null;
  }

  try {
    const base64Credentials = authHeader.slice(6);
    const credentials = atob(base64Credentials);
    const separatorIndex = credentials.indexOf(":");
    
    if (separatorIndex === -1) {
      return null;
    }

    return {
      username: credentials.slice(0, separatorIndex),
      password: credentials.slice(separatorIndex + 1),
    };
  } catch {
    return null;
  }
}

function createUnauthorizedResponse(): NextResponse {
  return new NextResponse("Authentication required", {
    status: 401,
    headers: {
      "WWW-Authenticate": `Basic realm="${REALM}", charset="UTF-8"`,
      "Content-Type": "text/plain",
    },
  });
}

export function middleware(request: NextRequest): NextResponse {
  const expectedUsername = process.env.BASIC_AUTH_USER;
  const expectedPassword = process.env.BASIC_AUTH_PASSWORD;

  if (!expectedUsername || !expectedPassword) {
    return NextResponse.next();
  }

  const authHeader = request.headers.get("authorization");
  const credentials = parseBasicAuth(authHeader);

  if (!credentials) {
    return createUnauthorizedResponse();
  }

  const isValidUsername = constantTimeCompare(
    credentials.username,
    expectedUsername
  );
  const isValidPassword = constantTimeCompare(
    credentials.password,
    expectedPassword
  );

  if (!isValidUsername || !isValidPassword) {
    return createUnauthorizedResponse();
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!api|_next/static|_next/image|assets|favicon\\.ico|robots\\.txt|sitemap\\.xml|manifest\\.webmanifest|vite\\.svg|.*\\.(?:css|js|map|png|jpg|jpeg|gif|svg|webp|ico|woff|woff2|ttf|eot|json)).*)",
  ],
};
