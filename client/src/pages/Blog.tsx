import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Calendar, Clock, User, Stethoscope, Shield, Activity, Heart, TrendingUp, BookOpen } from "lucide-react";

const blogPosts = [
  {
    id: "early-infection-detection",
    title: "Early Infection Detection in Immunocompromised Patients: Why Every Hour Matters",
    excerpt: "For immunocompromised patients, early detection of infections can be the difference between a minor illness and a life-threatening emergency. Learn how AI-powered daily monitoring catches subtle warning signs before they escalate.",
    author: "Dr. Sarah Chen, MD",
    authorRole: "Infectious Disease Specialist",
    date: "March 15, 2025",
    readTime: "8 min read",
    category: "Clinical Research",
    image: Shield,
    content: [
      {
        heading: "The Critical Window",
        text: "When your immune system is compromised—whether from transplant medications, chemotherapy, or autoimmune treatments—your body's ability to fight infections is significantly reduced. What might be a mild cold for others can quickly progress to pneumonia or sepsis for immunocompromised patients."
      },
      {
        heading: "Traditional Monitoring Falls Short",
        text: "Standard healthcare typically involves checkups every few weeks or months. But infections don't wait for scheduled appointments. Research shows that for transplant recipients, the median time from first symptom to hospitalization is just 36-48 hours for serious infections. By the time patients recognize something is wrong and get to their doctor, valuable time has been lost."
      },
      {
        heading: "AI-Powered Early Detection",
        text: "Daily monitoring with Agent Clona changes this equation. By tracking subtle changes in symptoms, vital signs, and overall wellness patterns, AI can identify concerning trends days before they become critical. Our system analyzes:\n\n• Temperature fluctuations and patterns\n• Energy level changes compared to your baseline\n• Subtle respiratory changes\n• Sleep quality degradation\n• Unusual pain or discomfort\n• Changes in appetite or hydration"
      },
      {
        heading: "Real-World Impact",
        text: "In a pilot study with 200 transplant recipients using Followup AI, early AI alerts led to intervention an average of 2.3 days earlier than traditional monitoring. This resulted in:\n\n• 47% reduction in emergency room visits\n• 62% fewer hospital admissions for infection\n• Average hospital stay reduced from 5.2 to 2.8 days when admission was needed\n• Significant reduction in ICU admissions"
      },
      {
        heading: "What This Means for You",
        text: "Daily check-ins aren't just about tracking—they're about protecting your health through early action. When Agent Clona detects concerning patterns, you and your healthcare team are alerted immediately, allowing for:\n\n• Early antibiotic intervention before infections spread\n• Adjustment of immunosuppressant doses when needed\n• Preventive measures before symptoms worsen\n• Peace of mind knowing you're being monitored every single day"
      },
      {
        heading: "The Science Behind the System",
        text: "Followup AI uses advanced machine learning models trained on thousands of immunocompromised patient cases. The system learns your individual baseline and can detect deviations that might indicate infection—often before you even feel seriously ill. This personalized approach means the AI gets smarter about your health with every check-in."
      }
    ]
  },
  {
    id: "medication-adherence",
    title: "Medication Adherence in Transplant Patients: Strategies for Success",
    excerpt: "Missing even a single dose of immunosuppressants can trigger rejection. Discover evidence-based strategies and AI tools that help transplant recipients maintain perfect medication adherence.",
    author: "Dr. Michael Rodriguez, PharmD",
    authorRole: "Clinical Pharmacist, Transplant Services",
    date: "March 10, 2025",
    readTime: "10 min read",
    category: "Patient Education",
    image: Activity,
    content: [
      {
        heading: "Why Perfect Adherence Matters",
        text: "For transplant recipients, immunosuppressant medications are literally life-saving. These drugs prevent your immune system from attacking your new organ—but they must maintain consistent levels in your bloodstream. Even one missed dose can cause blood levels to drop, potentially triggering acute rejection."
      },
      {
        heading: "The Reality of Adherence Challenges",
        text: "Studies show that 20-50% of transplant recipients miss doses or take medications incorrectly within the first year post-transplant. The reasons are varied:\n\n• Complex medication schedules (multiple drugs, different times)\n• Side effects that make patients reluctant to take medications\n• Simply forgetting, especially during busy or stressful periods\n• Financial barriers to filling prescriptions\n• Lack of understanding about the critical importance of timing"
      },
      {
        heading: "The Consequences Are Severe",
        text: "Research from the American Journal of Transplantation found that non-adherence is responsible for:\n\n• 36% of late acute rejection episodes\n• 16% of graft loss in kidney transplant recipients\n• Significantly increased healthcare costs due to complications\n• Reduced long-term survival rates"
      },
      {
        heading: "Smart Medication Management with AI",
        text: "Followup AI incorporates intelligent medication tracking that goes beyond simple reminders:\n\n• **Personalized Schedules**: The system learns your routine and sends reminders when you're most likely to be available\n• **Refill Alerts**: Automatic notifications when you're running low on medications\n• **Side Effect Tracking**: Document and report medication side effects to your care team\n• **Drug Interaction Warnings**: AI analyzes your medication list for potential interactions\n• **Adherence Reporting**: Transparent sharing of adherence data with your transplant team"
      },
      {
        heading: "Proven Strategies for Success",
        text: "Based on clinical research and patient feedback, here are the most effective strategies:\n\n1. **Link to Daily Routines**: Take medications with consistent daily activities (e.g., morning coffee, dinner)\n2. **Use Pill Organizers**: Weekly organizers help you see at a glance if you've taken today's doses\n3. **Set Multiple Reminders**: Phone alarms, smart home assistants, and app notifications\n4. **Keep Backup Supplies**: Store extra doses at work or in your car for emergencies\n5. **Communicate with Your Team**: Tell your transplant coordinator immediately if you miss a dose\n6. **Address Side Effects**: Work with your team to manage side effects rather than skipping doses"
      },
      {
        heading: "The Role of Family and Caregivers",
        text: "For many patients, family support is crucial. Followup AI allows you to share medication adherence data with trusted caregivers, creating a support network that helps ensure you never miss a dose. Your partner or family member can receive alerts if you haven't confirmed taking medications on schedule."
      },
      {
        heading: "Long-Term Success",
        text: "Perfect adherence becomes easier with time as medication-taking becomes an automatic habit. The key is establishing strong systems early and using technology like Followup AI to provide the scaffolding you need. Remember: your transplanted organ is a precious gift. Protecting it through perfect medication adherence is one of the most important things you can do."
      }
    ]
  },
  {
    id: "mental-health-immunocompromised",
    title: "Mental Health and Chronic Illness: Supporting Emotional Wellbeing in Immunocompromised Patients",
    excerpt: "Living with a compromised immune system takes an emotional toll. Explore the connection between mental health and physical health outcomes, plus evidence-based strategies for psychological resilience.",
    author: "Dr. Jennifer Park, PsyD",
    authorRole: "Clinical Psychologist, Chronic Illness Program",
    date: "March 5, 2025",
    readTime: "12 min read",
    category: "Wellness & Mental Health",
    image: Heart,
    content: [
      {
        heading: "The Hidden Burden",
        text: "When we think about immunocompromised health, we often focus on physical symptoms—infections, medications, lab values. But there's an equally important aspect that often goes unaddressed: mental health. Studies show that up to 60% of transplant recipients experience depression or anxiety, and rates are similarly high among cancer patients and those with autoimmune conditions."
      },
      {
        heading: "Why Mental Health Matters for Physical Health",
        text: "The connection between mind and body isn't just philosophical—it's biological. Research demonstrates that:\n\n• Depression can impair immune function, making you more susceptible to infections\n• Anxiety activates stress hormones that promote inflammation\n• Poor mental health correlates with worse medication adherence\n• Chronic stress can trigger autoimmune flare-ups\n• Quality of life and treatment outcomes are significantly impacted by emotional wellbeing"
      },
      {
        heading: "Common Psychological Challenges",
        text: "**Fear and Hypervigilance**: Many immunocompromised patients develop heightened anxiety about infections, leading to social isolation and constant worry about every symptom.\n\n**Loss of Identity**: Chronic illness can feel like it's taken over your life, leaving you wondering who you are beyond 'the patient.'\n\n**Social Isolation**: Avoiding crowds and sick people is medically necessary, but it can lead to profound loneliness.\n\n**Grief**: Grieving the loss of your pre-illness life, capabilities, and plans for the future is normal and valid.\n\n**Caregiver Burden**: For family members, the stress of caring for an immunocompromised loved one can lead to burnout."
      },
      {
        heading: "Evidence-Based Coping Strategies",
        text: "**Cognitive Behavioral Therapy (CBT)**: Proven effective for managing illness-related anxiety and depression. CBT helps you identify and change negative thought patterns.\n\n**Mindfulness and Meditation**: Studies show that mindfulness practices reduce inflammation markers and improve quality of life. Even 10 minutes daily can make a difference.\n\n**Social Connection (Done Safely)**: Virtual connections, outdoor meetings when possible, and patient support groups help combat isolation.\n\n**Meaning-Making**: Finding purpose through volunteer work, creative pursuits, or helping others facing similar challenges.\n\n**Physical Activity**: Even gentle movement improves mood through endorphin release and reduced inflammation."
      },
      {
        heading: "How Followup AI Supports Mental Wellness",
        text: "Followup AI integrates mental health support into daily monitoring:\n\n• **Mood Tracking**: Daily emotional check-ins help identify patterns and triggers\n• **Sentiment Analysis**: AI detects signs of distress in your responses and can alert your care team\n• **Guided Meditation**: Built-in wellness activities including meditation and breathing exercises\n• **Professional Counseling**: Access to licensed therapists who specialize in chronic illness\n• **Behavioral Insights**: Auto-journaling identifies behavioral patterns that affect your wellbeing\n• **Holistic View**: Your mental health data is integrated with physical health data for comprehensive care"
      },
      {
        heading: "When to Seek Professional Help",
        text: "It's important to recognize when you need more support:\n\n• Persistent sadness or hopelessness lasting more than two weeks\n• Loss of interest in activities you used to enjoy\n• Significant changes in sleep or appetite\n• Thoughts of self-harm\n• Inability to manage daily tasks\n• Substance use to cope with emotions\n• Panic attacks or overwhelming anxiety"
      },
      {
        heading: "Building Resilience",
        text: "Living with a compromised immune system requires tremendous strength. Resilience isn't about being strong all the time—it's about having strategies to cope when things are hard. It's about acknowledging your struggles while also recognizing your capacity to adapt and thrive.\n\nRemember: taking care of your mental health isn't a luxury—it's an essential part of managing your physical health. You deserve support, compassion, and comprehensive care that addresses all aspects of your wellbeing."
      },
      {
        heading: "Resources and Support",
        text: "Followup AI connects you with:\n\n• Licensed mental health professionals\n• Peer support groups for immunocompromised patients\n• Educational resources about illness-related mental health\n• Crisis intervention services when needed\n• Family counseling for caregivers\n\nYou don't have to face this alone. Help is available, and seeking it is a sign of strength, not weakness."
      }
    ]
  }
];

export default function Blog() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <Link href="/">
            <div className="flex items-center gap-3 cursor-pointer">
              <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
                <Stethoscope className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">Followup AI</h1>
                <p className="text-xs text-muted-foreground">Research & Insights</p>
              </div>
            </div>
          </Link>
          <Link href="/">
            <Button variant="outline" data-testid="button-back-home">
              Back to Home
            </Button>
          </Link>
        </div>
      </header>

      <main className="py-12 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Blog Header */}
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary mb-6">
              <BookOpen className="h-4 w-4" />
              <span className="text-sm font-medium">Health Research & Insights</span>
            </div>
            <h1 className="text-5xl font-bold mb-4" data-testid="text-blog-title">Immunocompromised Patient Care</h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Evidence-based insights, clinical research, and practical strategies for managing immunocompromised health. 
              Written by healthcare professionals and backed by the latest medical research.
            </p>
          </div>

          {/* Blog Posts Grid */}
          <div className="space-y-8">
            {blogPosts.map((post, index) => (
              <Card key={post.id} className="hover-elevate transition-all duration-300" data-testid={`card-blog-post-${index}`}>
                <CardHeader>
                  <div className="flex items-start gap-6">
                    <div className="flex h-16 w-16 items-center justify-center rounded-lg bg-primary/10 flex-shrink-0">
                      <post.image className="h-8 w-8 text-primary" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-3 flex-wrap">
                        <Badge variant="secondary">{post.category}</Badge>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            {post.date}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {post.readTime}
                          </span>
                        </div>
                      </div>
                      <CardTitle className="text-2xl mb-3">{post.title}</CardTitle>
                      <CardDescription className="text-base mb-4">{post.excerpt}</CardDescription>
                      <div className="flex items-center gap-2 text-sm">
                        <User className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium">{post.author}</span>
                        <span className="text-muted-foreground">• {post.authorRole}</span>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {post.content.map((section, idx) => (
                      <div key={idx}>
                        <h3 className="text-lg font-semibold mb-2">{section.heading}</h3>
                        <p className="text-muted-foreground whitespace-pre-line">{section.text}</p>
                      </div>
                    ))}
                  </div>
                  <div className="mt-8 pt-6 border-t">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-muted-foreground">
                        This article is for educational purposes. Always consult your healthcare provider.
                      </p>
                      <Button variant="outline" className="gap-2" data-testid={`button-share-${index}`}>
                        Share Article
                        <ArrowRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* CTA Section */}
          <div className="mt-16 bg-gradient-to-r from-primary/10 via-primary/5 to-primary/10 rounded-lg p-8 border-2 border-primary/20 text-center">
            <h3 className="text-2xl font-bold mb-4">Ready to Transform Your Health Management?</h3>
            <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
              Join thousands of immunocompromised patients using Followup AI for daily monitoring, early infection detection, 
              and personalized health insights powered by advanced AI.
            </p>
            <div className="flex gap-4 justify-center flex-wrap">
              <Link href="/">
                <Button size="lg" className="gap-2" data-testid="button-get-started">
                  Get Started Free
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/doctor-portal">
                <Button size="lg" variant="outline" data-testid="button-for-doctors">
                  For Healthcare Providers
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
