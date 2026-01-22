import { Link } from "wouter";
import { Stethoscope } from "lucide-react";
import { SiX, SiDiscord, SiLinkedin } from "react-icons/si";
import footerContent from "@/pages/content/footerContent.json";

const socialIcons: Record<string, typeof SiX> = {
  x: SiX,
  discord: SiDiscord,
  linkedin: SiLinkedin,
};

export function Footer() {
  const { resources, company, social } = footerContent;

  return (
    <footer className="bg-muted/50 border-t" data-testid="footer">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12">
          <div className="lg:col-span-1">
            <Link href="/">
              <div className="flex items-center gap-3 cursor-pointer mb-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
                  <Stethoscope className="h-6 w-6" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">Followup AI</h2>
                  <p className="text-xs text-muted-foreground">HIPAA-Compliant Health Platform</p>
                </div>
              </div>
            </Link>
            <p className="text-sm text-muted-foreground max-w-xs">
              Automated follow-up and early deterioration detection for chronic care patients.
            </p>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Resources</h3>
            <ul className="space-y-3">
              {Object.entries(resources).map(([key, item]) => (
                <li key={key}>
                  <Link href={item.path}>
                    <span
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                      title={item.short}
                      data-testid={`footer-link-${key}`}
                    >
                      {item.title}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Company</h3>
            <ul className="space-y-3">
              {Object.entries(company).map(([key, item]) => (
                <li key={key}>
                  <Link href={item.path}>
                    <span
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                      title={item.short}
                      data-testid={`footer-link-${key}`}
                    >
                      {item.title}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Connect</h3>
            <div className="flex gap-4">
              {Object.entries(social).map(([key, item]) => {
                const Icon = socialIcons[key];
                return (
                  <a
                    key={key}
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    title={item.short}
                    className="flex h-10 w-10 items-center justify-center rounded-md bg-muted hover:bg-accent transition-colors"
                    data-testid={`footer-social-${key}`}
                  >
                    {Icon && <Icon className="h-5 w-5" />}
                  </a>
                );
              })}
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              Email:{" "}
              <a
                href="mailto:admin@followupai.io"
                className="hover:text-foreground transition-colors"
                data-testid="footer-email"
              >
                admin@followupai.io
              </a>
            </p>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} Followup AI — All rights reserved.
          </p>
          <div className="flex gap-6">
            <Link href="/terms">
              <span className="text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
                Terms
              </span>
            </Link>
            <Link href="/privacy">
              <span className="text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
                Privacy
              </span>
            </Link>
            <Link href="/hipaa">
              <span className="text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
                HIPAA
              </span>
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
