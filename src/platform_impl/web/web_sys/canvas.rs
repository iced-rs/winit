use std::cell::Cell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use js_sys::{Array, Function, Reflect};
use smol_str::SmolStr;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    CompositionEvent, CssStyleDeclaration, Document, Event, EventTarget, FocusEvent,
    HtmlCanvasElement, KeyboardEvent, PointerEvent, WheelEvent,
};

use crate::dpi::{LogicalPosition, PhysicalPosition, PhysicalSize};
use crate::error::OsError as RootOE;
use crate::event::{Force, InnerSizeWriter, MouseButton, MouseScrollDelta};
use crate::keyboard::{Key, KeyLocation, ModifiersState, PhysicalKey};
use crate::platform_impl::OsError;
use crate::window::{WindowAttributes, WindowId as RootWindowId};

use super::super::cursor::CursorHandler;
use super::super::main_thread::MainThreadMarker;
use super::super::WindowId;
use super::animation_frame::AnimationFrameHandler;
use super::event_handle::EventListenerHandle;
use super::intersection_handle::IntersectionObserverHandle;
use super::media_query_handle::MediaQueryListHandle;
use super::pointer::PointerHandler;
use super::{event, fullscreen, ButtonsState, ResizeScaleHandle};

#[allow(dead_code)]
pub struct Canvas {
    common: Common,
    id: WindowId,
    pub has_focus: Rc<Cell<bool>>,
    pub prevent_default: Rc<Cell<bool>>,
    pub composing: Rc<Cell<bool>>,
    pub is_intersecting: Option<bool>,
    on_touch_start: Option<EventListenerHandle<dyn FnMut(Event)>>,
    on_focus: Option<EventListenerHandle<dyn FnMut(FocusEvent)>>,
    on_blur: Option<EventListenerHandle<dyn FnMut(FocusEvent)>>,
    on_keyboard_release: Option<EventListenerHandle<dyn FnMut(KeyboardEvent)>>,
    on_keyboard_press: Option<EventListenerHandle<dyn FnMut(KeyboardEvent)>>,
    on_mouse_wheel: Option<EventListenerHandle<dyn FnMut(WheelEvent)>>,
    on_dark_mode: Option<MediaQueryListHandle>,
    pointer_handler: PointerHandler,
    on_resize_scale: Option<ResizeScaleHandle>,
    on_intersect: Option<IntersectionObserverHandle>,
    animation_frame_handler: AnimationFrameHandler,
    on_touch_end: Option<EventListenerHandle<dyn FnMut(Event)>>,
    on_context_menu: Option<EventListenerHandle<dyn FnMut(PointerEvent)>>,
    on_composition_start: Option<EventListenerHandle<dyn FnMut(CompositionEvent)>>,
    on_composition_end: Option<EventListenerHandle<dyn FnMut(CompositionEvent)>>,
    on_text_update: Option<EventListenerHandle<dyn FnMut(Event)>>,
    pub cursor: CursorHandler,
}

pub struct Common {
    pub window: web_sys::Window,
    pub document: Document,
    /// Note: resizing the HTMLCanvasElement should go through `backend::set_canvas_size` to ensure
    /// the DPI factor is maintained. Note: this is read-only because we use a pointer to this
    /// for [`WindowHandle`][rwh_06::WindowHandle].
    raw: Rc<HtmlCanvasElement>,
    /// Owned `EditContext` instance, if the browser supports it.
    /// When `Some`, IME events are routed through
    /// it instead of through canvas keyboard events, which lets
    /// the OS IME composition (CJK, Indic, etc.) hand committed
    /// text into a wgpu canvas. `None` on non-Chromium browsers
    /// — IME is unavailable there.
    raw_edit_context: Option<Rc<JsValue>>,
    style: Style,
    old_size: Rc<Cell<PhysicalSize<u32>>>,
    current_size: Rc<Cell<PhysicalSize<u32>>>,
}

#[derive(Clone, Debug)]
pub struct Style {
    read: CssStyleDeclaration,
    write: CssStyleDeclaration,
}

impl Canvas {
    pub(crate) fn create(
        main_thread: MainThreadMarker,
        id: WindowId,
        window: web_sys::Window,
        document: Document,
        attr: &mut WindowAttributes,
    ) -> Result<Self, RootOE> {
        let canvas = match attr.platform_specific.canvas.take().map(|canvas| {
            Arc::try_unwrap(canvas)
                .map(|canvas| canvas.into_inner(main_thread))
                .unwrap_or_else(|canvas| canvas.get(main_thread).clone())
        }) {
            Some(canvas) => canvas,
            None => document
                .create_element("canvas")
                .map_err(|_| os_error!(OsError("Failed to create canvas element".to_owned())))?
                .unchecked_into(),
        };

        if attr.platform_specific.append && !document.contains(Some(&canvas)) {
            document
                .body()
                .expect("Failed to get body from document")
                .append_child(&canvas)
                .expect("Failed to append canvas to body");
        }

        // A tabindex is needed in order to capture local keyboard events.
        // A "0" value means that the element should be focusable in
        // sequential keyboard navigation, but its order is defined by the
        // document's source order.
        // https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/tabindex
        if attr.platform_specific.focusable {
            canvas
                .set_attribute("tabindex", "0")
                .map_err(|_| os_error!(OsError("Failed to set a tabindex".to_owned())))?;
        }

        let style = Style::new(&window, &canvas);

        let cursor = CursorHandler::new(main_thread, canvas.clone(), style.clone());

        // Optional EditContext for IME, for browser that support it. On other browsers,
        // IME is simply not available.
        let edit_context = Reflect::get(&window, &JsValue::from_str("EditContext"))
            .ok()
            .filter(|v| !v.is_undefined())
            .and_then(|ctor| {
                let ctor: &Function = ctor.unchecked_ref();
                Reflect::construct(ctor, &Array::new()).ok()
            });

        let common = Common {
            window: window.clone(),
            document: document.clone(),
            raw: Rc::new(canvas.clone()),
            raw_edit_context: edit_context.map(Rc::new),
            style,
            old_size: Rc::default(),
            current_size: Rc::default(),
        };

        if let Some(size) = attr.inner_size {
            let size = size.to_logical(super::scale_factor(&common.window));
            super::set_canvas_size(&common.document, &common.raw, &common.style, size);
        }

        if let Some(size) = attr.min_inner_size {
            let size = size.to_logical(super::scale_factor(&common.window));
            super::set_canvas_min_size(&common.document, &common.raw, &common.style, Some(size));
        }

        if let Some(size) = attr.max_inner_size {
            let size = size.to_logical(super::scale_factor(&common.window));
            super::set_canvas_max_size(&common.document, &common.raw, &common.style, Some(size));
        }

        if let Some(position) = attr.position {
            let position = position.to_logical(super::scale_factor(&common.window));
            super::set_canvas_position(&common.document, &common.raw, &common.style, position);
        }

        if attr.fullscreen.is_some() {
            fullscreen::request_fullscreen(&document, &canvas);
        }

        if attr.active {
            let _ = common.raw.focus();
        }

        Ok(Canvas {
            common,
            id,
            has_focus: Rc::new(Cell::new(false)),
            prevent_default: Rc::new(Cell::new(attr.platform_specific.prevent_default)),
            composing: Rc::new(Cell::new(false)),
            is_intersecting: None,
            on_touch_start: None,
            on_blur: None,
            on_focus: None,
            on_keyboard_release: None,
            on_keyboard_press: None,
            on_mouse_wheel: None,
            on_dark_mode: None,
            pointer_handler: PointerHandler::new(),
            on_resize_scale: None,
            on_intersect: None,
            animation_frame_handler: AnimationFrameHandler::new(window),
            on_touch_end: None,
            on_context_menu: None,
            on_composition_start: None,
            on_composition_end: None,
            on_text_update: None,
            cursor,
        })
    }

    pub fn set_cursor_lock(&self, lock: bool) -> Result<(), RootOE> {
        if lock {
            self.raw().request_pointer_lock();
        } else {
            self.common.document.exit_pointer_lock();
        }
        Ok(())
    }

    pub fn set_attribute(&self, attribute: &str, value: &str) {
        self.common
            .raw
            .set_attribute(attribute, value)
            .unwrap_or_else(|err| panic!("error: {err:?}\nSet attribute: {attribute}"))
    }

    pub fn position(&self) -> LogicalPosition<f64> {
        let bounds = self.common.raw.get_bounding_client_rect();
        let mut position = LogicalPosition { x: bounds.x(), y: bounds.y() };

        if self.document().contains(Some(self.raw())) && self.style().get("display") != "none" {
            position.x += super::style_size_property(self.style(), "border-left-width")
                + super::style_size_property(self.style(), "padding-left");
            position.y += super::style_size_property(self.style(), "border-top-width")
                + super::style_size_property(self.style(), "padding-top");
        }

        position
    }

    #[inline]
    pub fn old_size(&self) -> PhysicalSize<u32> {
        self.common.old_size.get()
    }

    #[inline]
    pub fn inner_size(&self) -> PhysicalSize<u32> {
        self.common.current_size.get()
    }

    #[inline]
    pub fn set_old_size(&self, size: PhysicalSize<u32>) {
        self.common.old_size.set(size)
    }

    #[inline]
    pub fn set_current_size(&self, size: PhysicalSize<u32>) {
        self.common.current_size.set(size)
    }

    #[inline]
    pub fn window(&self) -> &web_sys::Window {
        &self.common.window
    }

    #[inline]
    pub fn document(&self) -> &Document {
        &self.common.document
    }

    #[inline]
    pub fn raw(&self) -> &HtmlCanvasElement {
        &self.common.raw
    }

    #[inline]
    pub fn style(&self) -> &Style {
        &self.common.style
    }

    pub fn on_touch_start(&mut self) {
        let prevent_default = Rc::clone(&self.prevent_default);
        self.on_touch_start = Some(self.common.add_event("touchstart", move |event: Event| {
            if prevent_default.get() {
                event.prevent_default();
            }
        }));
    }

    pub fn on_blur<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(),
    {
        self.on_blur = Some(self.common.add_event("blur", move |_: FocusEvent| {
            handler();
        }));
    }

    pub fn on_focus<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(),
    {
        self.on_focus = Some(self.common.add_event("focus", move |_: FocusEvent| {
            handler();
        }));
    }

    pub fn on_keyboard_release<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(PhysicalKey, Key, Option<SmolStr>, KeyLocation, bool, ModifiersState),
    {
        let prevent_default = Rc::clone(&self.prevent_default);
        let composing = Rc::clone(&self.composing);
        self.on_keyboard_release =
            Some(self.common.add_event("keyup", move |event: KeyboardEvent| {
                if composing.get() || is_ime_composing(&event) {
                    return;
                }
                if prevent_default.get() {
                    event.prevent_default();
                }
                let key = event::key(&event);
                let modifiers = event::keyboard_modifiers(&event);
                handler(
                    event::key_code(&event),
                    key,
                    event::key_text(&event),
                    event::key_location(&event),
                    event.repeat(),
                    modifiers,
                );
            }));
    }

    pub fn on_keyboard_press<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(PhysicalKey, Key, Option<SmolStr>, KeyLocation, bool, ModifiersState),
    {
        let prevent_default = Rc::clone(&self.prevent_default);
        let composing = Rc::clone(&self.composing);
        self.on_keyboard_press =
            Some(self.common.add_event("keydown", move |event: KeyboardEvent| {
                let suppress = composing.get() || is_ime_composing(&event);
                if suppress {
                    return;
                }
                if prevent_default.get() {
                    event.prevent_default();
                }
                let key = event::key(&event);
                let modifiers = event::keyboard_modifiers(&event);
                handler(
                    event::key_code(&event),
                    key,
                    event::key_text(&event),
                    event::key_location(&event),
                    event.repeat(),
                    modifiers,
                );
            }));
    }

    pub fn on_cursor_leave<F>(&mut self, handler: F)
    where
        F: 'static + FnMut(ModifiersState, Option<i32>),
    {
        self.pointer_handler.on_cursor_leave(&self.common, handler)
    }

    pub fn on_cursor_enter<F>(&mut self, handler: F)
    where
        F: 'static + FnMut(ModifiersState, Option<i32>),
    {
        self.pointer_handler.on_cursor_enter(&self.common, handler)
    }

    pub fn on_mouse_release<M, T>(&mut self, mouse_handler: M, touch_handler: T)
    where
        M: 'static + FnMut(ModifiersState, i32, PhysicalPosition<f64>, MouseButton),
        T: 'static + FnMut(ModifiersState, i32, PhysicalPosition<f64>, Force),
    {
        self.pointer_handler.on_mouse_release(&self.common, mouse_handler, touch_handler)
    }

    pub fn on_mouse_press<M, T>(&mut self, mouse_handler: M, touch_handler: T)
    where
        M: 'static + FnMut(ModifiersState, i32, PhysicalPosition<f64>, MouseButton),
        T: 'static + FnMut(ModifiersState, i32, PhysicalPosition<f64>, Force),
    {
        self.pointer_handler.on_mouse_press(
            &self.common,
            mouse_handler,
            touch_handler,
            Rc::clone(&self.prevent_default),
        )
    }

    pub fn on_cursor_move<M, T, B>(&mut self, mouse_handler: M, touch_handler: T, button_handler: B)
    where
        M: 'static + FnMut(ModifiersState, i32, &mut dyn Iterator<Item = PhysicalPosition<f64>>),
        T: 'static
            + FnMut(ModifiersState, i32, &mut dyn Iterator<Item = (PhysicalPosition<f64>, Force)>),
        B: 'static + FnMut(ModifiersState, i32, PhysicalPosition<f64>, ButtonsState, MouseButton),
    {
        self.pointer_handler.on_cursor_move(
            &self.common,
            mouse_handler,
            touch_handler,
            button_handler,
            Rc::clone(&self.prevent_default),
        )
    }

    pub fn on_touch_cancel<F>(&mut self, handler: F)
    where
        F: 'static + FnMut(i32, PhysicalPosition<f64>, Force),
    {
        self.pointer_handler.on_touch_cancel(&self.common, handler)
    }

    pub fn on_mouse_wheel<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(i32, MouseScrollDelta, ModifiersState),
    {
        let window = self.common.window.clone();
        let prevent_default = Rc::clone(&self.prevent_default);
        self.on_mouse_wheel = Some(self.common.add_event("wheel", move |event: WheelEvent| {
            if prevent_default.get() {
                event.prevent_default();
            }

            if let Some(delta) = event::mouse_scroll_delta(&window, &event) {
                let modifiers = event::mouse_modifiers(&event);
                handler(0, delta, modifiers);
            }
        }));
    }

    pub fn on_dark_mode<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(bool),
    {
        self.on_dark_mode = Some(MediaQueryListHandle::new(
            &self.common.window,
            "(prefers-color-scheme: dark)",
            move |mql| handler(mql.matches()),
        ));
    }

    pub(crate) fn on_resize_scale<S, R>(&mut self, scale_handler: S, size_handler: R)
    where
        S: 'static + Fn(PhysicalSize<u32>, f64),
        R: 'static + Fn(PhysicalSize<u32>),
    {
        self.on_resize_scale = Some(ResizeScaleHandle::new(
            self.window().clone(),
            self.document().clone(),
            self.raw().clone(),
            self.style().clone(),
            scale_handler,
            size_handler,
        ));
    }

    pub(crate) fn on_intersection<F>(&mut self, handler: F)
    where
        F: 'static + FnMut(bool),
    {
        self.on_intersect = Some(IntersectionObserverHandle::new(self.raw(), handler));
    }

    pub(crate) fn on_animation_frame<F>(&mut self, f: F)
    where
        F: 'static + FnMut(),
    {
        self.animation_frame_handler.on_animation_frame(f)
    }

    pub(crate) fn on_context_menu(&mut self) {
        let prevent_default = Rc::clone(&self.prevent_default);
        self.on_context_menu =
            Some(self.common.add_event("contextmenu", move |event: PointerEvent| {
                if prevent_default.get() {
                    event.prevent_default();
                }
            }));
    }

    pub(crate) fn on_composition_start<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(Option<String>, Option<(usize, usize)>),
    {
        let prevent_default = Rc::clone(&self.prevent_default);
        let composing = Rc::clone(&self.composing);
        self.on_composition_start =
            self.common.add_ime_event("compositionstart", move |event: CompositionEvent| {
                composing.set(true);
                if prevent_default.get() {
                    event.prevent_default();
                }
                handler(event.data(), None);
            });
    }

    pub(crate) fn on_composition_end<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(Option<String>),
    {
        let prevent_default = Rc::clone(&self.prevent_default);
        let composing = Rc::clone(&self.composing);
        self.on_composition_end =
            self.common.add_ime_event("compositionend", move |event: CompositionEvent| {
                composing.set(false);
                if prevent_default.get() {
                    event.prevent_default();
                }
                handler(event.data());
            });
    }

    pub(crate) fn on_text_update<F>(&mut self, mut handler: F)
    where
        F: 'static + FnMut(Option<String>, Option<(usize, usize)>),
    {
        let prevent_default = Rc::clone(&self.prevent_default);
        self.on_text_update = self.common.add_ime_event("textupdate", move |event: Event| {
            if prevent_default.get() {
                event.prevent_default();
            }
            let text_update_event = JsValue::from(event);
            let text = Reflect::get(&text_update_event, &JsValue::from_str("text"))
                .ok()
                .and_then(|v| v.as_string());
            handler(text, None);
        });
    }

    pub(crate) fn is_support_edit_context(&self) -> bool {
        self.common.raw_edit_context.is_some()
    }

    pub(crate) fn enable_edit_context(&self) {
        if let Some(raw_edit_context) = &self.common.raw_edit_context {
            let canvas_js = JsValue::from(self.common.raw.deref());
            let _ = Reflect::set(&canvas_js, &JsValue::from_str("editContext"), raw_edit_context);
        }
    }

    pub(crate) fn disable_edit_context(&self) {
        if self.common.raw_edit_context.is_none() {
            return;
        }
        let canvas_js = JsValue::from(self.common.raw.deref());
        let _ = Reflect::set(&canvas_js, &JsValue::from_str("editContext"), &JsValue::NULL);
    }

    pub(crate) fn update_character_bounds(&self, x: f64, y: f64, width: f64, height: f64) {
        const BOUNDS_RANGE: usize = 64;

        let Some(edit_context) = self.common.raw_edit_context.as_ref() else {
            return;
        };
        let canvas_rect = self.common.raw.get_bounding_client_rect();
        let vp_x = canvas_rect.x() + x;
        let vp_y = canvas_rect.y() + y;
        let edit_context_js: &JsValue = edit_context.deref();

        // Chromium positions the IME candidate window primarily off
        // `updateSelectionBounds`, then `updateControlBounds`, then
        // (only if both are unset) the per-character bounds set via
        // `updateCharacterBounds`. We push all three so the candidate
        // window lands next to the caret regardless of which one the
        // browser version prefers.
        let caret_rect = match web_sys::DomRect::new_with_x_and_y_and_width_and_height(
            vp_x, vp_y, width, height,
        ) {
            Ok(rect) => rect,
            Err(_) => return,
        };

        if let Ok(f) = Reflect::get(edit_context_js, &JsValue::from_str("updateSelectionBounds")) {
            let f: &Function = f.unchecked_ref();
            let _ = f.call1(edit_context_js, &caret_rect);
        }

        let control_rect = match web_sys::DomRect::new_with_x_and_y_and_width_and_height(
            canvas_rect.x(),
            canvas_rect.y(),
            canvas_rect.width(),
            canvas_rect.height(),
        ) {
            Ok(rect) => rect,
            Err(_) => return,
        };
        if let Ok(f) = Reflect::get(edit_context_js, &JsValue::from_str("updateControlBounds")) {
            let f: &Function = f.unchecked_ref();
            let _ = f.call1(edit_context_js, &control_rect);
        }

        let bounds = Array::new();
        for _ in 0..BOUNDS_RANGE {
            let Ok(rect) =
                web_sys::DomRect::new_with_x_and_y_and_width_and_height(vp_x, vp_y, width, height)
            else {
                return;
            };
            bounds.push(&rect);
        }
        let Ok(update_fn) =
            Reflect::get(edit_context_js, &JsValue::from_str("updateCharacterBounds"))
        else {
            return;
        };
        let update_fn: &Function = update_fn.unchecked_ref();
        let _ = update_fn.call2(edit_context_js, &JsValue::from_f64(0.0), &bounds);
    }

    pub fn request_fullscreen(&self) {
        fullscreen::request_fullscreen(self.document(), self.raw());
    }

    pub fn exit_fullscreen(&self) {
        fullscreen::exit_fullscreen(self.document(), self.raw());
    }

    pub fn is_fullscreen(&self) -> bool {
        fullscreen::is_fullscreen(self.document(), self.raw())
    }

    pub fn request_animation_frame(&self) {
        self.animation_frame_handler.request();
    }

    pub(crate) fn handle_scale_change(
        &self,
        runner: &super::super::event_loop::runner::Shared,
        event_handler: impl FnOnce(crate::event::Event<()>),
        current_size: PhysicalSize<u32>,
        scale: f64,
    ) {
        // First, we send the `ScaleFactorChanged` event:
        self.set_current_size(current_size);
        let new_size = {
            let new_size = Arc::new(Mutex::new(current_size));
            event_handler(crate::event::Event::WindowEvent {
                window_id: RootWindowId(self.id),
                event: crate::event::WindowEvent::ScaleFactorChanged {
                    scale_factor: scale,
                    inner_size_writer: InnerSizeWriter::new(Arc::downgrade(&new_size)),
                },
            });

            let new_size = *new_size.lock().unwrap();
            new_size
        };

        if current_size != new_size {
            // Then we resize the canvas to the new size, a new
            // `Resized` event will be sent by the `ResizeObserver`:
            let new_size = new_size.to_logical(scale);
            super::set_canvas_size(self.document(), self.raw(), self.style(), new_size);

            // Set the size might not trigger the event because the calculation is inaccurate.
            self.on_resize_scale
                .as_ref()
                .expect("expected Window to still be active")
                .notify_resize();
        } else if self.old_size() != new_size {
            // Then we at least send a resized event.
            self.set_old_size(new_size);
            runner.send_event(crate::event::Event::WindowEvent {
                window_id: RootWindowId(self.id),
                event: crate::event::WindowEvent::Resized(new_size),
            })
        }
    }

    pub fn remove_listeners(&mut self) {
        self.on_touch_start = None;
        self.on_focus = None;
        self.on_blur = None;
        self.on_keyboard_release = None;
        self.on_keyboard_press = None;
        self.on_mouse_wheel = None;
        self.on_dark_mode = None;
        self.pointer_handler.remove_listeners();
        self.on_resize_scale = None;
        self.on_intersect = None;
        self.animation_frame_handler.cancel();
        self.on_touch_end = None;
        self.on_context_menu = None;
        self.on_composition_start = None;
        self.on_composition_end = None;
        self.on_text_update = None;
    }
}

impl Common {
    pub fn add_event<E, F>(
        &self,
        event_name: &'static str,
        handler: F,
    ) -> EventListenerHandle<dyn FnMut(E)>
    where
        E: 'static + AsRef<web_sys::Event> + wasm_bindgen::convert::FromWasmAbi,
        F: 'static + FnMut(E),
    {
        EventListenerHandle::new(self.raw.deref().clone(), event_name, Closure::new(handler))
    }

    pub fn add_ime_event<E, F>(
        &self,
        event_name: &'static str,
        handler: F,
    ) -> Option<EventListenerHandle<dyn FnMut(E)>>
    where
        E: 'static + AsRef<web_sys::Event> + wasm_bindgen::convert::FromWasmAbi,
        F: 'static + FnMut(E),
    {
        let edit_context = self.raw_edit_context.as_ref()?.deref().clone();
        let target: EventTarget = edit_context.unchecked_into();
        Some(EventListenerHandle::new(target, event_name, Closure::new(handler)))
    }

    pub fn raw(&self) -> &HtmlCanvasElement {
        &self.raw
    }
}

impl Style {
    fn new(window: &web_sys::Window, canvas: &HtmlCanvasElement) -> Self {
        #[allow(clippy::disallowed_methods)]
        let read = window
            .get_computed_style(canvas)
            .expect("Failed to obtain computed style")
            // this can't fail: we aren't using a pseudo-element
            .expect("Invalid pseudo-element");

        #[allow(clippy::disallowed_methods)]
        let write = canvas.style();

        Self { read, write }
    }

    pub(crate) fn get(&self, property: &str) -> String {
        self.read.get_property_value(property).expect("Invalid property")
    }

    pub(crate) fn remove(&self, property: &str) {
        self.write.remove_property(property).expect("Property is read only");
    }

    pub(crate) fn set(&self, property: &str, value: &str) {
        self.write.set_property(property, value).expect("Property is read only");
    }
}

fn is_ime_composing(event: &KeyboardEvent) -> bool {
    event.is_composing() || event.key_code() == 229
}
