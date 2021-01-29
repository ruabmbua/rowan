use std::{
    cell::Cell,
    fmt,
    hash::{Hash, Hasher},
    iter,
    mem::ManuallyDrop,
    ptr,
};

use countme::Count;

use crate::{
    green::{GreenElementRef, GreenNodeData, SyntaxKind},
    sll,
    utility_types::Delta,
    Direction, GreenNode, GreenToken, NodeOrToken, SyntaxText, TextRange, TextSize, TokenAtOffset,
    WalkEvent,
};

pub struct SyntaxNode {
    ptr: ptr::NonNull<NodeData>,
}

impl Clone for SyntaxNode {
    #[inline]
    fn clone(&self) -> Self {
        let rc = match self.data().rc.get().checked_add(1) {
            Some(it) => it,
            None => std::process::abort(),
        };
        self.data().rc.set(rc);
        SyntaxNode { ptr: self.ptr }
    }
}

impl Drop for SyntaxNode {
    #[inline]
    fn drop(&mut self) {
        let rc = self.data().rc.get() - 1;
        self.data().rc.set(rc);
        if rc == 0 {
            unsafe { free(self.ptr) }
        }
    }
}

#[inline(never)]
unsafe fn free(mut data: ptr::NonNull<NodeData>) {
    loop {
        debug_assert_eq!(data.as_ref().rc.get(), 0);
        let node = Box::from_raw(data.as_ptr());
        match node.parent.take() {
            Some(parent) => {
                if node.mutable {
                    sll::unlink(&parent.as_ref().first, &*node)
                }
                let rc = parent.as_ref().rc.get() - 1;
                parent.as_ref().rc.set(rc);
                if rc == 0 {
                    data = parent;
                } else {
                    break;
                }
            }
            None => {
                let _ = GreenNode::from_raw(node.green.get());
                break;
            }
        }
    }
}

// Identity semantics for hash & eq
impl PartialEq for SyntaxNode {
    #[inline]
    fn eq(&self, other: &SyntaxNode) -> bool {
        self.key() == other.key()
    }
}

impl Eq for SyntaxNode {}

impl Hash for SyntaxNode {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl fmt::Debug for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyntaxNode")
            .field("kind", &self.kind())
            .field("text_range", &self.text_range())
            .finish()
    }
}

impl fmt::Display for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.preorder_with_tokens()
            .filter_map(|event| match event {
                WalkEvent::Enter(NodeOrToken::Token(token)) => Some(token),
                _ => None,
            })
            .try_for_each(|it| fmt::Display::fmt(&it, f))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyntaxToken {
    parent: SyntaxNode,
    index: u32,
    offset: TextSize,
}

impl fmt::Display for SyntaxToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.text(), f)
    }
}

pub type SyntaxElement = NodeOrToken<SyntaxNode, SyntaxToken>;

impl From<SyntaxNode> for SyntaxElement {
    #[inline]
    fn from(node: SyntaxNode) -> SyntaxElement {
        NodeOrToken::Node(node)
    }
}

impl From<SyntaxToken> for SyntaxElement {
    #[inline]
    fn from(token: SyntaxToken) -> SyntaxElement {
        NodeOrToken::Token(token)
    }
}

impl fmt::Display for SyntaxElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeOrToken::Node(it) => fmt::Display::fmt(it, f),
            NodeOrToken::Token(it) => fmt::Display::fmt(it, f),
        }
    }
}

struct NodeData {
    _c: Count<SyntaxNode>,
    rc: Cell<u32>,
    parent: Cell<Option<ptr::NonNull<NodeData>>>,
    index: Cell<u32>,
    offset: TextSize,
    green: Cell<ptr::NonNull<GreenNodeData>>,

    mutable: bool,
    // The following links only have meaning when state is `Mut`.
    first: Cell<*const NodeData>,
    // Invariant: never null
    next: Cell<*const NodeData>,
    prev: Cell<*const NodeData>,
}

unsafe impl sll::Elem for NodeData {
    fn prev(&self) -> &Cell<*const Self> {
        &self.prev
    }

    fn next(&self) -> &Cell<*const Self> {
        &self.next
    }

    fn key(&self) -> &Cell<u32> {
        &self.index
    }
}

impl NodeData {
    #[inline]
    fn new(
        parent: Option<SyntaxNode>,
        index: u32,
        offset: TextSize,
        green: ptr::NonNull<GreenNodeData>,
        mutable: bool,
    ) -> ptr::NonNull<NodeData> {
        let res = NodeData {
            _c: Count::new(),
            rc: Cell::new(1),
            parent: {
                let parent = ManuallyDrop::new(parent);
                Cell::new(parent.as_ref().map(|it| it.ptr))
            },
            index: Cell::new(index),
            offset,
            green: Cell::new(green),

            mutable,
            first: Cell::new(ptr::null()),
            next: Cell::new(ptr::null()),
            prev: Cell::new(ptr::null()),
        };
        unsafe {
            let mut res = Box::into_raw(Box::new(res));
            if mutable {
                if let Err(node) = sll::init((*res).parent().map(|it| &it.first), &*res) {
                    Box::from_raw(res);
                    res = node as *mut _;
                }
            }
            ptr::NonNull::new_unchecked(res)
        }
    }

    fn to_owned(&self) -> SyntaxNode {
        let rc = self.rc.get();
        self.rc.set(rc + 1);
        SyntaxNode { ptr: ptr::NonNull::from(self) }
    }

    #[inline]
    fn parent(&self) -> Option<&NodeData> {
        self.parent.get().map(|it| unsafe { &*it.as_ptr() })
    }
    #[inline]
    fn green(&self) -> &GreenNodeData {
        unsafe { &*self.green.get().as_ptr() }
    }
    #[inline]
    fn index(&self) -> u32 {
        self.index.get()
    }

    #[inline]
    fn offset(&self) -> TextSize {
        if self.mutable {
            self.offset_mut()
        } else {
            self.offset
        }
    }

    #[cold]
    fn offset_mut(&self) -> TextSize {
        match self.parent() {
            Some(parent) => {
                let rel_offset =
                    parent.green().children().raw.nth(self.index() as usize).unwrap().rel_offset();
                parent.offset_mut() + rel_offset
            }
            None => 0.into(),
        }
    }

    fn detach(&self) {
        assert!(self.mutable);
        let parent = match self.parent.take() {
            Some(parent) => parent,
            None => return,
        };

        unsafe {
            sll::adjust(self, self.index() + 1, Delta::Sub(1));
            let mut node = parent.as_ref();
            sll::unlink(&node.first, self);
            let _ = GreenNode::into_raw(self.green().to_owned());

            let mut green = node.green().remove_child(self.index() as usize);

            loop {
                match node.parent() {
                    Some(parent) => {
                        node.green.set((&*green).into());
                        green = parent.green().replace_child(node.index() as usize, green.into());
                        node = parent;
                    }
                    None => {
                        let _ = GreenNode::from_raw(node.green.get());
                        node.green.set(GreenNode::into_raw(green));
                        break;
                    }
                }
            }
        }
    }
}

impl SyntaxNode {
    pub fn new_root(green: GreenNode) -> SyntaxNode {
        SyntaxNode { ptr: NodeData::new(None, 0, 0.into(), GreenNode::into_raw(green), false) }
    }

    pub fn new_root_mut(green: GreenNode) -> SyntaxNode {
        SyntaxNode { ptr: NodeData::new(None, 0, 0.into(), GreenNode::into_raw(green), true) }
    }

    fn new_child(
        green: &GreenNodeData,
        parent: SyntaxNode,
        index: u32,
        offset: TextSize,
    ) -> SyntaxNode {
        let mutable = parent.data().mutable;
        SyntaxNode { ptr: NodeData::new(Some(parent), index, offset, green.into(), mutable) }
    }

    pub fn clone_for_update(&self) -> SyntaxNode {
        assert!(!self.data().mutable);
        match self.parent() {
            Some(parent) => {
                let parent = parent.clone_for_update();
                SyntaxNode::new_child(self.green_ref(), parent, self.data().index(), self.offset())
            }
            None => SyntaxNode::new_root_mut(self.green_ref().to_owned()),
        }
    }

    fn key(&self) -> (ptr::NonNull<GreenNodeData>, TextSize) {
        (self.data().green.get(), self.offset())
    }

    #[inline]
    fn data(&self) -> &NodeData {
        unsafe { self.ptr.as_ref() }
    }

    pub fn replace_with(&self, replacement: GreenNode) -> GreenNode {
        assert_eq!(self.kind(), replacement.kind());
        match &self.parent() {
            None => replacement,
            Some(parent) => {
                let new_parent = parent
                    .green_ref()
                    .replace_child(self.data().index() as usize, replacement.into());
                parent.replace_with(new_parent)
            }
        }
    }

    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.green_ref().kind()
    }

    #[inline]
    fn offset(&self) -> TextSize {
        self.data().offset()
    }

    #[inline]
    pub fn text_range(&self) -> TextRange {
        let offset = self.offset();
        let len = self.green_ref().text_len();
        TextRange::at(offset, len)
    }

    #[inline]
    pub fn text(&self) -> SyntaxText {
        SyntaxText::new(self.clone())
    }

    #[inline]
    pub fn green(&self) -> GreenNode {
        self.green_ref().to_owned()
    }
    #[inline]
    fn green_ref(&self) -> &GreenNodeData {
        self.data().green()
    }

    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        self.data().parent().map(|it| it.to_owned())
    }

    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        iter::successors(Some(self.clone()), SyntaxNode::parent)
    }

    #[inline]
    pub fn children(&self) -> SyntaxNodeChildren {
        SyntaxNodeChildren::new(self.clone())
    }

    #[inline]
    pub fn children_with_tokens(&self) -> SyntaxElementChildren {
        SyntaxElementChildren::new(self.clone())
    }

    pub fn first_child(&self) -> Option<SyntaxNode> {
        self.green_ref().children().raw.enumerate().find_map(|(index, child)| {
            child.as_ref().into_node().map(|green| {
                SyntaxNode::new_child(
                    green,
                    self.clone(),
                    index as u32,
                    self.offset() + child.rel_offset(),
                )
            })
        })
    }
    pub fn last_child(&self) -> Option<SyntaxNode> {
        self.green_ref().children().raw.enumerate().rev().find_map(|(index, child)| {
            child.as_ref().into_node().map(|green| {
                SyntaxNode::new_child(
                    green,
                    self.clone(),
                    index as u32,
                    self.offset() + child.rel_offset(),
                )
            })
        })
    }

    pub fn first_child_or_token(&self) -> Option<SyntaxElement> {
        self.green_ref().children().raw.next().map(|child| {
            SyntaxElement::new(child.as_ref(), self.clone(), 0, self.offset() + child.rel_offset())
        })
    }
    pub fn last_child_or_token(&self) -> Option<SyntaxElement> {
        self.green_ref().children().raw.enumerate().next_back().map(|(index, child)| {
            SyntaxElement::new(
                child.as_ref(),
                self.clone(),
                index as u32,
                self.offset() + child.rel_offset(),
            )
        })
    }

    pub fn next_sibling(&self) -> Option<SyntaxNode> {
        let parent = self.data().parent()?;
        let mut children = parent.green().children().raw.enumerate();
        children.nth(self.data().index() as usize);
        children.find_map(|(index, child)| {
            child.as_ref().into_node().map(|green| {
                SyntaxNode::new_child(
                    green,
                    parent.to_owned(),
                    index as u32,
                    parent.offset() + child.rel_offset(),
                )
            })
        })
    }
    pub fn prev_sibling(&self) -> Option<SyntaxNode> {
        let parent = self.data().parent()?;
        let mut children = parent.green().children().raw.enumerate().rev();
        children.nth(parent.green().children().len() - self.data().index() as usize);
        children.find_map(|(index, child)| {
            child.as_ref().into_node().map(|green| {
                SyntaxNode::new_child(
                    green,
                    parent.to_owned(),
                    index as u32,
                    parent.offset() + child.rel_offset(),
                )
            })
        })
    }

    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        let parent = self.data().parent()?;
        parent.green().children().raw.enumerate().nth(self.data().index() as usize + 1).map(
            |(index, child)| {
                SyntaxElement::new(
                    child.as_ref(),
                    parent.to_owned(),
                    index as u32,
                    parent.offset() + child.rel_offset(),
                )
            },
        )
    }
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        let parent = self.data().parent()?;
        parent
            .green()
            .children()
            .raw
            .enumerate()
            .nth(self.data().index().checked_sub(1)? as usize)
            .map(|(index, child)| {
                SyntaxElement::new(
                    child.as_ref(),
                    parent.to_owned(),
                    index as u32,
                    parent.offset() + child.rel_offset(),
                )
            })
    }

    pub fn first_token(&self) -> Option<SyntaxToken> {
        self.first_child_or_token()?.first_token()
    }
    pub fn last_token(&self) -> Option<SyntaxToken> {
        self.last_child_or_token()?.last_token()
    }

    #[inline]
    pub fn siblings(&self, direction: Direction) -> impl Iterator<Item = SyntaxNode> {
        iter::successors(Some(self.clone()), move |node| match direction {
            Direction::Next => node.next_sibling(),
            Direction::Prev => node.prev_sibling(),
        })
    }

    #[inline]
    pub fn siblings_with_tokens(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = SyntaxElement> {
        let me: SyntaxElement = self.clone().into();
        iter::successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }

    #[inline]
    pub fn descendants(&self) -> impl Iterator<Item = SyntaxNode> {
        self.preorder().filter_map(|event| match event {
            WalkEvent::Enter(node) => Some(node),
            WalkEvent::Leave(_) => None,
        })
    }

    #[inline]
    pub fn descendants_with_tokens(&self) -> impl Iterator<Item = SyntaxElement> {
        self.preorder_with_tokens().filter_map(|event| match event {
            WalkEvent::Enter(it) => Some(it),
            WalkEvent::Leave(_) => None,
        })
    }

    #[inline]
    pub fn preorder(&self) -> Preorder {
        Preorder::new(self.clone())
    }

    #[inline]
    pub fn preorder_with_tokens<'a>(&'a self) -> impl Iterator<Item = WalkEvent<SyntaxElement>> {
        let start: SyntaxElement = self.clone().into();
        iter::successors(Some(WalkEvent::Enter(start.clone())), move |pos| {
            let next = match pos {
                WalkEvent::Enter(el) => match el {
                    NodeOrToken::Node(node) => match node.first_child_or_token() {
                        Some(child) => WalkEvent::Enter(child),
                        None => WalkEvent::Leave(node.clone().into()),
                    },
                    NodeOrToken::Token(token) => WalkEvent::Leave(token.clone().into()),
                },
                WalkEvent::Leave(el) => {
                    if el == &start {
                        return None;
                    }
                    match el.next_sibling_or_token() {
                        Some(sibling) => WalkEvent::Enter(sibling),
                        None => WalkEvent::Leave(el.parent().unwrap().into()),
                    }
                }
            };
            Some(next)
        })
    }

    pub fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        // TODO: this could be faster if we first drill-down to node, and only
        // then switch to token search. We should also replace explicit
        // recursion with a loop.
        let range = self.text_range();
        assert!(
            range.start() <= offset && offset <= range.end(),
            "Bad offset: range {:?} offset {:?}",
            range,
            offset
        );
        if range.is_empty() {
            return TokenAtOffset::None;
        }

        let mut children = self.children_with_tokens().filter(|child| {
            let child_range = child.text_range();
            !child_range.is_empty()
                && (child_range.start() <= offset && offset <= child_range.end())
        });

        let left = children.next().unwrap();
        let right = children.next();
        assert!(children.next().is_none());

        if let Some(right) = right {
            match (left.token_at_offset(offset), right.token_at_offset(offset)) {
                (TokenAtOffset::Single(left), TokenAtOffset::Single(right)) => {
                    TokenAtOffset::Between(left, right)
                }
                _ => unreachable!(),
            }
        } else {
            left.token_at_offset(offset)
        }
    }

    pub fn covering_element(&self, range: TextRange) -> SyntaxElement {
        let mut res: SyntaxElement = self.clone().into();
        loop {
            assert!(
                res.text_range().contains_range(range),
                "Bad range: node range {:?}, range {:?}",
                res.text_range(),
                range,
            );
            res = match &res {
                NodeOrToken::Token(_) => return res,
                NodeOrToken::Node(node) => match node.child_or_token_at_range(range) {
                    Some(it) => it,
                    None => return res,
                },
            };
        }
    }

    pub fn child_or_token_at_range(&self, range: TextRange) -> Option<SyntaxElement> {
        let rel_range = range - self.offset();
        self.green_ref().child_at_range(rel_range).map(|(index, rel_offset, green)| {
            SyntaxElement::new(green, self.clone(), index as u32, self.offset() + rel_offset)
        })
    }

    pub fn detach(&self) {
        self.data().detach()
    }
}

impl SyntaxToken {
    fn new(parent: SyntaxNode, index: u32, offset: TextSize) -> SyntaxToken {
        SyntaxToken { parent, index, offset }
    }

    pub fn replace_with(&self, replacement: GreenToken) -> GreenNode {
        assert_eq!(self.kind(), replacement.kind());
        let mut replacement = Some(replacement);
        let parent = self.parent();
        let me = self.index;

        let children = parent.green_ref().children().enumerate().map(|(i, child)| {
            if i as u32 == me {
                replacement.take().unwrap().into()
            } else {
                child.cloned()
            }
        });
        let new_parent = GreenNode::new(parent.kind(), children);
        parent.replace_with(new_parent)
    }

    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.green().kind()
    }

    #[inline]
    pub fn text_range(&self) -> TextRange {
        TextRange::at(self.offset, self.green().text_len())
    }

    #[inline]
    pub fn text(&self) -> &str {
        self.green().text()
    }

    #[inline]
    pub fn green(&self) -> &GreenToken {
        self.parent.green_ref().children().nth(self.index as usize).unwrap().as_token().unwrap()
    }

    #[inline]
    pub fn parent(&self) -> SyntaxNode {
        self.parent.clone()
    }

    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        self.parent().ancestors()
    }

    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.parent.green_ref().children().raw.enumerate().nth((self.index + 1) as usize).map(
            |(index, child)| {
                SyntaxElement::new(
                    child.as_ref(),
                    self.parent(),
                    index as u32,
                    self.parent.offset() + child.rel_offset(),
                )
            },
        )
    }
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.parent
            .green_ref()
            .children()
            .raw
            .enumerate()
            .nth(self.index.checked_sub(1)? as usize)
            .map(|(index, child)| {
                SyntaxElement::new(
                    child.as_ref(),
                    self.parent(),
                    index as u32,
                    self.parent.offset() + child.rel_offset(),
                )
            })
    }

    pub fn siblings_with_tokens(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = SyntaxElement> {
        let me: SyntaxElement = self.clone().into();
        iter::successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }

    pub fn next_token(&self) -> Option<SyntaxToken> {
        match self.next_sibling_or_token() {
            Some(element) => element.first_token(),
            None => self
                .parent()
                .ancestors()
                .find_map(|it| it.next_sibling_or_token())
                .and_then(|element| element.first_token()),
        }
    }
    pub fn prev_token(&self) -> Option<SyntaxToken> {
        match self.prev_sibling_or_token() {
            Some(element) => element.last_token(),
            None => self
                .parent()
                .ancestors()
                .find_map(|it| it.prev_sibling_or_token())
                .and_then(|element| element.last_token()),
        }
    }
}

impl SyntaxElement {
    fn new(
        element: GreenElementRef<'_>,
        parent: SyntaxNode,
        index: u32,
        offset: TextSize,
    ) -> SyntaxElement {
        match element {
            NodeOrToken::Node(node) => {
                SyntaxNode::new_child(node, parent, index as u32, offset).into()
            }
            NodeOrToken::Token(_) => SyntaxToken::new(parent, index as u32, offset).into(),
        }
    }

    #[inline]
    pub fn text_range(&self) -> TextRange {
        match self {
            NodeOrToken::Node(it) => it.text_range(),
            NodeOrToken::Token(it) => it.text_range(),
        }
    }

    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        match self {
            NodeOrToken::Node(it) => it.kind(),
            NodeOrToken::Token(it) => it.kind(),
        }
    }

    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        match self {
            NodeOrToken::Node(it) => it.parent(),
            NodeOrToken::Token(it) => Some(it.parent()),
        }
    }

    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        match self {
            NodeOrToken::Node(it) => it.ancestors(),
            NodeOrToken::Token(it) => it.parent().ancestors(),
        }
    }

    pub fn first_token(&self) -> Option<SyntaxToken> {
        match self {
            NodeOrToken::Node(it) => it.first_token(),
            NodeOrToken::Token(it) => Some(it.clone()),
        }
    }
    pub fn last_token(&self) -> Option<SyntaxToken> {
        match self {
            NodeOrToken::Node(it) => it.last_token(),
            NodeOrToken::Token(it) => Some(it.clone()),
        }
    }

    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            NodeOrToken::Node(it) => it.next_sibling_or_token(),
            NodeOrToken::Token(it) => it.next_sibling_or_token(),
        }
    }
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            NodeOrToken::Node(it) => it.prev_sibling_or_token(),
            NodeOrToken::Token(it) => it.prev_sibling_or_token(),
        }
    }

    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        assert!(self.text_range().start() <= offset && offset <= self.text_range().end());
        match self {
            NodeOrToken::Token(token) => TokenAtOffset::Single(token.clone()),
            NodeOrToken::Node(node) => node.token_at_offset(offset),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxNodeChildren {
    next: Option<SyntaxNode>,
}

impl SyntaxNodeChildren {
    fn new(parent: SyntaxNode) -> SyntaxNodeChildren {
        SyntaxNodeChildren { next: parent.first_child() }
    }
}

impl Iterator for SyntaxNodeChildren {
    type Item = SyntaxNode;
    fn next(&mut self) -> Option<SyntaxNode> {
        self.next.take().map(|next| {
            self.next = next.next_sibling();
            next
        })
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxElementChildren {
    next: Option<SyntaxElement>,
}

impl SyntaxElementChildren {
    fn new(parent: SyntaxNode) -> SyntaxElementChildren {
        SyntaxElementChildren { next: parent.first_child_or_token() }
    }
}

impl Iterator for SyntaxElementChildren {
    type Item = SyntaxElement;
    fn next(&mut self) -> Option<SyntaxElement> {
        self.next.take().map(|next| {
            self.next = next.next_sibling_or_token();
            next
        })
    }
}

pub struct Preorder {
    root: SyntaxNode,
    next: Option<WalkEvent<SyntaxNode>>,
    skip_subtree: bool,
}

impl Preorder {
    fn new(root: SyntaxNode) -> Preorder {
        let next = Some(WalkEvent::Enter(root.clone()));
        Preorder { root, next, skip_subtree: false }
    }

    pub fn skip_subtree(&mut self) {
        self.skip_subtree = true;
    }
    #[cold]
    fn do_skip(&mut self) {
        self.next = self.next.take().map(|next| match next {
            WalkEvent::Enter(first_child) => WalkEvent::Leave(first_child.parent().unwrap()),
            WalkEvent::Leave(parent) => WalkEvent::Leave(parent),
        })
    }
}

impl Iterator for Preorder {
    type Item = WalkEvent<SyntaxNode>;

    fn next(&mut self) -> Option<WalkEvent<SyntaxNode>> {
        if self.skip_subtree {
            self.do_skip();
            self.skip_subtree = false;
        }
        let next = self.next.take();
        self.next = next.as_ref().and_then(|next| {
            Some(match next {
                WalkEvent::Enter(node) => match node.first_child() {
                    Some(child) => WalkEvent::Enter(child),
                    None => WalkEvent::Leave(node.clone()),
                },
                WalkEvent::Leave(node) => {
                    if node == &self.root {
                        return None;
                    }
                    match node.next_sibling() {
                        Some(sibling) => WalkEvent::Enter(sibling),
                        None => WalkEvent::Leave(node.parent().unwrap()),
                    }
                }
            })
        });
        next
    }
}
