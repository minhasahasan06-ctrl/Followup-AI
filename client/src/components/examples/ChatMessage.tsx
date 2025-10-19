import { ChatMessage } from "../ChatMessage";

export default function ChatMessageExample() {
  return (
    <div className="p-4 max-w-4xl">
      <ChatMessage
        role="user"
        content="I've been feeling more fatigued than usual lately, and my joints are a bit achy. Should I be concerned?"
        timestamp="2 mins ago"
      />
      <ChatMessage
        role="assistant"
        content="I understand your concern. Fatigue and joint achiness can be common for immunocompromised patients. Let me help you assess this. Have you noticed any other symptoms like fever, swelling, or changes in your medication routine?"
        timestamp="2 mins ago"
        isGP={true}
        entities={[
          { text: "Fatigue", type: "symptom" },
          { text: "Joint pain", type: "symptom" },
        ]}
      />
      <ChatMessage
        role="user"
        content="No fever, but there's some mild swelling in my hands. I've been taking my Prednisone as prescribed."
        timestamp="1 min ago"
      />
      <ChatMessage
        role="assistant"
        content="Thank you for that information. The mild swelling combined with fatigue could indicate inflammation. Since you're on Prednisone, I'd recommend monitoring these symptoms closely. Consider taking an OTC anti-inflammatory like Ibuprofen 200mg if needed. However, let's schedule a follow-up with your doctor if symptoms persist beyond 48 hours."
        timestamp="Just now"
        isGP={true}
        entities={[
          { text: "Swelling", type: "symptom" },
          { text: "Prednisone", type: "medication" },
          { text: "Ibuprofen", type: "medication" },
          { text: "200mg", type: "dosage" },
        ]}
      />
    </div>
  );
}
