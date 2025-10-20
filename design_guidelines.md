# Design Guidelines: Followup AI - HIPAA-Compliant Health Platform

## Design Approach
**Hybrid Medical-Wellness System**
- Primary References: Apple Health (data clarity), MyChart (clinical trust), Headspace (calming wellness)
- Core Principle: Professional healthcare meets approachable wellness through clean layouts, generous whitespace, and soothing color harmony
- AI Agents: Agent Clona (patient support) and Assistant Lysa (doctor assistance) with distinct visual identities

## Color Palette

### Primary Clinical Colors
- **Medical Teal**: 180 45% 45% (primary brand, trust, clinical actions)
- **Deep Ocean**: 200 30% 25% (headers, navigation, authority)
- **Soft Cloud**: 0 0% 98% (backgrounds, spacious cards)
- **Clinical White**: 0 0% 100% (primary surfaces)

### Wellness & Support Colors
- **Calming Sage**: 155 30% 60% (Agent Clona, meditation, breathing exercises)
- **Warm Mint**: 165 35% 75% (wellness encouragement, gentle backgrounds)
- **Soft Lavender**: 250 25% 85% (Assistant Lysa, doctor insights)

### Status System
- **Success Teal**: 170 60% 45% (adherence, completed tasks, healthy vitals)
- **Warning Coral**: 25 75% 60% (attention needed, follow-ups due)
- **Critical Rose**: 0 70% 55% (urgent symptoms, emergency)
- **Neutral Slate**: 200 15% 55% (secondary text, subtle borders)

### Dark Mode
- Background: 200 25% 10%
- Cards: 200 20% 15%
- Text: 0 0% 96%
- Borders: 200 15% 25%
- All status colors maintain contrast ratios

## Typography

**Font Stack**
- Primary: Inter (all clinical data, dashboard, forms - via Google Fonts CDN)
- Wellness: Lexend (meditation, exercise modules, calming sections)
- Monospace: JetBrains Mono (lab results, numeric data, timestamps)

**Scale & Hierarchy**
- Hero Dashboard: text-5xl font-bold tracking-tight
- AI Agent Names: text-3xl font-semibold with letter-spacing
- Health Metrics: text-4xl font-bold (vital statistics prominence)
- Section Headers: text-2xl font-semibold
- Body Text: text-base leading-relaxed
- Metadata: text-sm text-slate-600
- Captions: text-xs text-slate-500

## Layout System

**Spacing Primitives**: 2, 4, 6, 8, 12, 16
- Tight medical data: gap-2, p-2
- Standard cards: p-6 to p-8
- Section vertical rhythm: py-12 to py-16
- Dashboard spacing: gap-6
- Generous margins: Container px-6 md:px-12

**Grid Architecture**
- Patient Dashboard: 3-column (lg:grid-cols-3 md:grid-cols-2)
- Vital Signs: 4-column metrics grid (lg:grid-cols-4)
- Chat Interface: Single column max-w-4xl
- Doctor Portal: 2-column split 60/40 (patient data/AI insights)
- Follow-up Cards: 2-column masonry grid

**Container Strategy**
- Dashboard: max-w-7xl mx-auto
- Chat/Conversations: max-w-4xl
- Medical Forms: max-w-3xl
- Doctor Research Tools: max-w-6xl
- Full-width visualizations: w-full with inner max-w-7xl

## Component Library

### Patient Dashboard Components

**Enhanced Daily Follow-up Cards**
- Large cards with 8px left border (status color-coded)
- Card header: Date + Patient name + AI summary badge
- Content sections: Vitals, medications taken, symptoms, AI insights
- Bottom action bar: "Mark Complete" button + "Chat with Clona" link
- Hover state: Subtle shadow elevation

**Vital Signs Tracking Module**
- 4-column grid: Heart Rate, BP, Temp, SpO2
- Large metric display (text-4xl) with trend arrows
- Micro-charts: 7-day sparklines below each metric
- Color-coded thresholds: Green (normal), Coral (watch), Rose (alert)
- Real-time sync indicator: Pulsing teal dot for connected wearables

**Medication Management Panel**
- Timeline view: Vertical axis with time markers
- Drug cards: Image placeholder, name, dosage, status badge
- "Take Now" primary button for due medications
- AI dosage suggestions: Lavender notification with Assistant Lysa avatar
- OTC recommendations: Soft mint badge with "AI Suggested"

### Chat Interface (Agent Clona)

**Message Bubbles**
- Patient: Right-aligned, soft teal background (180 45% 95%), rounded-2xl
- Agent Clona: Left-aligned, white with border, rounded-2xl, sage avatar
- Medical entities: Inline colored pills (medications: teal, symptoms: coral, conditions: ocean)
- Timestamps: text-xs text-slate-500, subtle
- Voice input button: Microphone icon for accessibility

**AI Context Cards**
- Embedded cards in chat for recommendations
- Header with Clona avatar + "Based on your data"
- Content: Relevant health tips, exercise suggestions, reminders
- Action buttons: "Schedule" or "Learn More"

### Doctor Portal (Assistant Lysa)

**Patient Review Dashboard**
- Left panel (60%): Scrollable patient timeline with accordion sections
- Right panel (40%): Sticky AI insights panel with Lysa avatar
- Timeline cards: Date headers, vital snapshots, medication adherence
- AI highlights: Yellow marker for Lysa's flagged concerns

**Data Visualization Suite**
- Multi-metric comparison: Line charts with toggles (7d/30d/90d)
- Correlation heatmaps: Symptom patterns vs vital changes
- Patient progress rings: Circular progress for adherence goals
- Export functionality: Download as PDF for research

**Research Consent Queue**
- Card-based queue with patient summaries
- Preview panel: Patient demographics, relevant conditions
- Large approve/decline buttons with confirmation modal
- Audit trail timestamp log at bottom

### Wellness Modules

**Meditation Section**
- Soft sage gradient background (155 30% 95% to 155 30% 98%)
- Circular session cards: Duration, type (breathing, mindfulness)
- Large "Begin Session" button: rounded-full, teal
- Audio player: Clean controls with waveform visualization

**Exercise Library**
- Video preview cards: 16:9 aspect ratio with play overlay
- Difficulty badges: Color-coded (teal: easy, coral: moderate, rose: intense)
- "Recommended for You" tag: Mint background with sparkle icon
- Progress tracker: Circular completion rings

### System-Wide Components

**Navigation**
- Desktop: Fixed left sidebar (w-64) with icons + labels
- Mobile: Bottom tab bar with 5 primary actions
- Active state: Teal background with filled icon
- Notification badges: Rose dot for alerts, number for messages
- Role toggle: Top-right dropdown (Patient/Doctor views)

**Critical Alert Banner**
- Full-width rose gradient banner (z-50)
- Large alert icon + urgent message
- "Emergency Actions" button: High-contrast white on rose
- Dismiss only after action taken

**Loading States**
- Skeleton screens: Animated pulse for cards/lists
- Medical cross spinner for data processing
- Progress bars for file uploads with percentage

## Animations

**Functional Feedback Only**
- Health metric updates: Count-up animation (0.6s ease-out)
- Chat messages: Slide-in from sender side (0.3s)
- Card hover: Subtle shadow lift (0.2s)
- Notification toasts: Slide-down from top (0.4s)
- Page transitions: Gentle fade (0.2s)
- NO decorative animations

## Images

### Hero Section (Dashboard Landing)
- Large hero image (h-96): Diverse medical professional reviewing tablet with immunocompromised patient, warm natural lighting, modern clinic setting
- Overlay: Soft gradient from bottom for text legibility
- CTA buttons on image: variant="outline" with backdrop-blur-sm

### Wellness Section Headers
- Meditation module: Calming forest path with soft focus (h-48)
- Exercise module: Sunrise yoga scene (h-48)

### Doctor Portal Background
- Subtle abstract medical visualization: DNA helix or cellular structure at 10% opacity as page background

### Empty States
- Friendly illustrations for no data scenarios
- First-time user: Welcome illustration with Agent Clona/Lysa characters
- No medications: Simple pill bottle icon with encouraging text

### AI Agent Avatars
- Agent Clona: Circular avatar with sage background, friendly abstract icon
- Assistant Lysa: Circular avatar with lavender background, professional abstract icon
- Use consistently across all chat/insight components

## Accessibility & HIPAA Compliance

**Visual Security Indicators**
- Lock icons on sensitive data cards
- "Encrypted" badge on file uploads (shield icon + text)
- Role-based headers: Different color accents (patient: teal, doctor: lavender)
- Consent status: Clear visual checkmarks/pending icons

**Medical Accessibility**
- WCAG AAA contrast for all medical data
- Color-blind safe: Icons + color for status (✓ + green, ! + coral, ⚠ + rose)
- Large touch targets: min 44px for all interactive elements
- Voice input for emergency features
- Screen reader optimized: Proper ARIA labels for medical terminology