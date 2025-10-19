# Design Guidelines: AI Health Assistant for Immunocompromised Patients

## Design Approach
**Hybrid Approach**: Medical Design System + Wellness App Aesthetics
- Primary inspiration: Apple Health (data clarity), MyChart (medical professionalism), Headspace (calming wellness sections)
- Design System: Custom healthcare system balancing clinical precision with patient comfort
- Core principle: Trust through clarity - every element must feel medically sound yet approachable

## Color Palette

### Primary Colors (Clinical Interface)
- **Primary Medical Blue**: 210 100% 45% (trust, professionalism)
- **Deep Navy**: 215 25% 20% (headers, critical text)
- **Soft White/Clinical**: 0 0% 98% (backgrounds, cards)

### Wellness Colors (Meditation/Exercise Sections)
- **Calming Sage**: 150 25% 55% (meditation, breathing exercises)
- **Warm Peach**: 25 70% 85% (wellness encouragement, gentle alerts)

### Status & Alert Colors
- **Success Green**: 145 65% 45% (medication adherence, completed tasks)
- **Warning Amber**: 35 85% 55% (health alerts, follow-up reminders)
- **Critical Red**: 0 75% 50% (emergency symptoms, urgent actions)
- **Neutral Gray**: 210 10% 60% (secondary text, borders)

### Dark Mode (HIPAA-compliant healthcare dark theme)
- Background: 215 25% 12%
- Cards: 215 20% 18%
- Text: 0 0% 95%
- Maintain color ratios for accessibility

## Typography

**Font Families**
- Primary: Inter (clinical clarity, excellent readability for health data)
- Wellness Sections: Lexend (gentle, calming for meditation/exercise)
- Medical Data: SF Mono / JetBrains Mono (test results, numerical data)

**Hierarchy**
- Hero/Dashboard: text-4xl font-semibold (medical professional authority)
- Section Headers: text-2xl font-semibold
- Health Metrics: text-3xl font-bold (vital statistics prominence)
- Body Text: text-base leading-relaxed (comfortable reading for lengthy medical info)
- Captions/Labels: text-sm text-gray-600 (subtle guidance)

## Layout System

**Spacing Primitives**: 2, 4, 8, 12, 16 (tailwind units)
- Tight medical data: p-2, gap-2
- Card padding: p-4 to p-6
- Section spacing: py-8 to py-12
- Dashboard gaps: gap-4 (compact information density)

**Grid Structure**
- Dashboard: 3-column grid (lg:grid-cols-3, md:grid-cols-2, grid-cols-1)
- Health metrics: 2-4 column responsive grids
- Chat interface: Single column max-w-4xl centered
- Doctor console: 2-column split (patient data left, diagnostic tools right)

**Container Max-widths**
- Chat/Conversation: max-w-4xl
- Dashboard: max-w-7xl
- Medical forms: max-w-2xl
- Full-width data visualizations: w-full with inner max-w-6xl

## Component Library

### Core Medical Components

**Patient Dashboard Card**
- White card with subtle shadow (shadow-md)
- Border-l-4 with status color coding (blue: stable, amber: attention needed, red: urgent)
- Health metric display: Large numbers (text-3xl) with unit labels (text-sm)
- Trend indicators: Small arrows with percentage changes

**Chat Interface**
- Patient messages: Right-aligned, soft blue background (210 100% 95%)
- AI responses: Left-aligned, white with border
- Medical entity highlights: Inline colored pills (medications: green, symptoms: amber, diagnoses: blue)
- Timestamp: text-xs text-gray-500

**Clinical Assessment Module**
- Camera viewfinder: Full-width with overlay guides
- Assessment results: Card grid showing detected conditions (anemia, jaundice, edema)
- Confidence scores: Progress bars with percentage
- Historical comparison: Side-by-side before/after thumbnails

**Medication Manager**
- Drug cards: Image placeholder, name (font-semibold), dosage, frequency
- OTC suggestion badge: Small green pill-shaped label
- AI dosage adjustment indicator: Subtle amber notification dot
- Schedule timeline: Vertical timeline with taken/missed states

**Wearable Data Integration**
- Real-time sync status indicator (pulsing green dot when active)
- Data source badges: Small device icons (Apple Watch, Fitbit, etc.)
- Graph visualizations: Line charts with 7-day/30-day toggles
- Metric cards: Heart rate, steps, sleep, SpO2 in 2x2 grid

**File Upload Zone**
- Dashed border drag-and-drop area
- Supported formats: Medical imaging (DICOM, JPEG), lab reports (PDF)
- Upload preview: Thumbnail grid with file names
- AI processing status: Loading spinner → Success checkmark

### Wellness Components

**Meditation/Mindfulness Section**
- Soft sage green background gradient
- Circular session cards with duration and type
- "Start Session" button: Rounded-full with calming hover states
- Guided audio player: Clean minimalist controls

**Exercise Module**
- Video demonstration preview (16:9 ratio)
- Difficulty badges: Easy/Moderate/Intense with color coding
- Personalized recommendations: "Recommended for your condition" tag
- Completion tracker: Circular progress rings

**Reminder System**
- Non-intrusive notification cards (top-right toast)
- Water intake: Glass icon with fill animation
- Exercise: Calendar icon with time
- Medication: Pill icon with urgency level

### Doctor Agent Interface

**Patient Review Dashboard**
- Split view: Patient timeline (left 60%) | AI insights (right 40%)
- Medical history accordion: Expandable sections by date
- AI suggestions panel: Highlighted recommendations with confidence scores
- Consultation notes: Rich text editor with medical autocomplete

**Doctor Verification Portal**
- Research consent requests: Card-based approval queue
- Patient data summary: Key metrics and AI-generated brief
- Approve/Reject actions: Large, distinct button pair
- Audit trail: Timestamped log of all verifications

### Emergency Features

**Critical Alert Banner**
- Full-width red banner at top (z-50)
- Large icon + urgent message
- "Call 911" button: Oversized, high-contrast
- Auto-scroll to top when triggered

**Symptom Checker Modal**
- Overlay with symptom input
- Real-time severity assessment
- Emergency escalation: Red zone triggers ambulance dispatch
- Non-urgent: Schedules doctor consultation

## Navigation

**Main Navigation**
- Sidebar: Fixed left navigation (desktop), bottom tabs (mobile)
- Icons: Clear medical symbols (dashboard, chat, wellness, settings)
- Active state: Filled icon + accent color background
- Badge notifications: Red dot for alerts, number for unread messages

**Role Switching**
- Toggle between Patient View / Doctor Agent View
- Visual indicator: Top-right user role badge
- Seamless context preservation

## Animations

**Use Sparingly**
- Health metric updates: Count-up number animation (fast, 0.5s)
- Chat message appearance: Gentle slide-in from appropriate side
- Notification toasts: Slide-in from top-right
- Loading states: Medical cross spinner or pulsing dot
- NO decorative animations - only functional feedback

## Images

**Hero Section**: Medical professional with immunocompromised patient (diverse, warm, reassuring) - NOT stock photo aesthetic, authentic healthcare setting
**Wellness Section Headers**: Calming nature scenes (meditation: forest, exercise: sunrise)
**Doctor Dashboard**: Abstract medical visualization (DNA helix, cellular imagery) as subtle background
**Empty States**: Friendly illustrations for no data yet (first-time user guidance)

## Accessibility

**HIPAA Visual Compliance**
- Lock icons on sensitive data cards
- "Encrypted" badges on file uploads
- Consent status clearly visible
- Role-based access visual indicators (different header colors)

**Medical Accessibility**
- High contrast for all medical data (WCAG AAA)
- Color-blind safe status indicators (icons + color)
- Large touch targets (min 44px) for immunocompromised patients with dexterity issues
- Clear medication instructions (large text, simple language)
- Emergency features: Voice-activated option for critical moments