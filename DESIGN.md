# Gemma 4 Design System

## Overview

Creative north star: "The Ethereal Intelligence".

Gemma 4 should feel monumental and weightless at the same time. The interface is not a chat toy. It is an editorial workspace where prompts, model state, and AI output live on an atmospheric canvas. Responses should feel like they emerge from the background instead of sitting inside generic boxed containers.

The target quality bar is intentional sophistication:

- Atmospheric UI over standard chat chrome
- Strong asymmetry between user and AI
- Light, blur, and surface hierarchy instead of visible section lines
- Calm density with clear technical affordances

## Visual Thesis

Mood: nocturnal, glassy, editorial, high-trust.

Material: layered obsidian surfaces with soft electric light.

Energy: precise, ambient, and quietly powerful.

## Layout Strategy

The app is organized into four zones:

1. Left history rail
2. Fixed top app bar with model selection
3. Central conversation stage
4. Floating command dock

Rules:

- No hard layout dividers for sectioning
- Use background shifts and 24px to 48px gaps instead of visible lines
- Keep the AI side wider than the user side
- Hide secondary controls in a flyout or secondary panel instead of cluttering the main stage

## Color System

Base tokens:

```text
background: #060e20
surface: #060e20
surface_container_low: #091328
surface_container: #0f1930
surface_container_high: #141f38
surface_container_highest: #192540
surface_variant: #192540
primary: #88adff
primary_container: #719eff
primary_dim: #0f6ef0
secondary: #c8d8f3
secondary_container: #38485d
tertiary: #fab0ff
tertiary_dim: #e48fed
on_surface: #dee5ff
on_surface_variant: #a3aac4
outline: #6d758c
outline_variant: #40485d
```

Rules:

- The "No-Line" rule is default: do not use visible 1px section borders
- Boundaries are created with surface shifts and spacing
- Floating UI may use a ghost boundary only when accessibility needs it
- Primary actions use a 135-degree gradient from `primary` to `primary_container`

## Typography

Typeface system:

- Display and headlines: Manrope
- Body and UI: Inter

Rules:

- Use Manrope for the greeting, product identity, and large headers
- Use Inter for prompts, responses, metadata, and controls
- Never use pure white text
- Use `on_surface` for user copy and `on_secondary_container` or softened text for AI body copy

## Elevation

Elevation is expressed through light and layered surfaces, not card borders.

Rules:

- Use recessed surfaces for structure
- Use soft ambient shadows with low opacity only for floating layers
- Use blur and transparency on the command dock and menus
- Keep elevated layers feeling integrated into the background

## Component Rules

### Top Bar

- Fixed
- Translucent
- Brand first
- Model dropdown lives here
- Technical controls stay compact

### History Rail

- Use `surface_container_low`
- Prioritize thread labels and calm navigation
- New chat button should feel integrated, not loud

### Conversation

- No horizontal separators between messages
- Use at least 32px vertical spacing between turns
- User messages align right and can be more compact
- AI messages align left and can occupy more width
- AI responses should read like editorial content, not stacked cards

### Command Dock

- Floating glass panel
- `surface_container_highest` at about 70% opacity
- 40px backdrop blur
- XL or full rounding
- Soft primary glow on focus

### Buttons

- Primary buttons use the Gemma gradient
- Secondary buttons are ghost buttons
- Avoid heavy fills for secondary actions

### Chips

- Use `secondary_container`
- Rounded medium to large
- Good for modalities, quick actions, and context labels

### Loading

- No spinner as the primary loading language
- Use a subtle shimmer bar at the top of the conversation stage

## UX Rules

- Model switching should feel deliberate and visible in the top bar
- Unsupported modalities must be disabled before submit
- Technical settings should live in a secondary flyout, not the main stage
- The first viewport should establish brand, model state, and task readiness immediately
- Empty state copy should be short and directional

## Interaction Modes

The app now supports two primary modes:

1. Text studio
2. Live vision call

Rules:

- Text studio remains the default editorial workspace
- Live vision call uses a full-stage camera surface with compact controls
- Live mode should prioritize the video feed, current model state, and rolling turns
- Fullscreen should be available from within live mode
- Voice output in live mode can use browser TTS when the model itself returns text

## Live Mode Notes

- Gemma 4 E2B and E4B accept text, image, and audio input
- Gemma 4 26B A4B and 31B accept text and image input only
- The current local Gemma instruction checkpoints generate text output, not native audio output
- Therefore, the live call UX should send microphone audio plus the current camera frame to Gemma, then read the text reply aloud using browser speech synthesis

## Gemma 4 Model Matrix

Verified against official Gemma 4 docs and model cards:

- Gemma 4 E2B: text, image, audio
- Gemma 4 E4B: text, image, audio
- Gemma 4 26B A4B MoE: text, image
- Gemma 4 31B: text, image

Audio input must be disabled for 26B A4B and 31B in the UI and rejected server-side.

## Implementation Notes

This repository should follow these implementation decisions:

- `DESIGN.md` is the design source of truth
- The top bar model selector is the primary switching mechanism
- The settings drawer contains system prompt, sampling controls, and machine telemetry
- The main stage remains focused on conversation and media context
- The interface should feel close to the supplied Gemma mockup without turning into a static replica
