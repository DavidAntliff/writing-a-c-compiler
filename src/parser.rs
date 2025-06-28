//! Parser for the C language
//!
//! Grammar:
//!   <program> ::= { <function-declaration> }
//!   <declaration> ::= <variable-declaration> | <function-declaration>
//!   <variable-declaration> ::= "int" <identifier> [ "=" <exp> ] ";"
//!   <function-declaration> ::= "int" <identifier> "(" <param-list> ")" ( <block> | ";" )
//!   <param-list> ::= "void" | "int" <identifier> { "," "int" <identifier> }
//!   <block> ::= "{" { <block-item> } "}"
//!   <block-item> ::= <statement> | <declaration>
//!   <for-init> ::= <variable-declaration> | [ <exp> ] ";"
//!   <statement> ::= [ <identifier> ":" ] <statement>
//!                 | "return" <exp> ";"
//!                 | <exp> ";"
//!                 | "if" "(" <exp> ")" <statement> [ "else" <statement> ]
//!                 | "goto" <identifier> ";"
//!                 | <block>
//!                 | "break" ";"
//!                 | "continue" ";"
//!                 | "while" "(" <exp> ")" <statement>
//!                 | "do" <statement> "while" "(" <exp> ")" ";"
//!                 | "for" "(" <for-init> [ <exp> ] ";" [ <exp> ] ")" <statement>
//!                 | ";"
//!   <exp> ::= <factor>
//!           | <exp> <binop> <exp>
//!           | <exp> "?" <exp> ":" <exp>
//!   <factor> ::= <int>
//!              | <identifier>
//!              | <unop> <factor>
//!              | "(" <exp> ")"
//!              | <identifier> "(" [ <argument-list> ] ")"
//!   <argument-list> ::= <exp> { "," <exp> }
//!   <unop> ::= "-" | "~" | "!"
//!   <binop> ::= "-" | "+" | "-" | "*" | "/" | "%"
//!               "&" | "|" | "^" | "<<" | ">>"
//!               "&&" | "||" | "==" | "!="
//!               "<" | "<=" | ">" | ">="
//!   <identifier> ::= ? An identifier token ?
//!   <int> ::= ? A constant token ?
//!

use crate::ast_c::{
    BinaryOperator, Block, BlockItem, Declaration, Expression, ForInit, FunDecl, Program,
    Statement, UnaryOperator, VarDecl,
};
use crate::lexer::{Identifier, Keyword, Token, TokenKind};
use thiserror::Error;
use winnow::combinator::{
    alt, cut_err, fail, opt, peek, preceded, repeat, repeat_till, separated, terminated, trace,
};
use winnow::dispatch;
use winnow::error::{StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::TokenSlice;
use winnow::token::{any, literal, take};

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct ParserError {
    pub message: String, // TODO: remove?
    pub expected: String,
    pub found: String,
    pub offset: usize,
}

impl ParserError {
    // Avoiding `From` so winnow types don't become part of our public API
    fn from_parse(
        error: winnow::error::ParseError<TokenSlice<'_, Token>, winnow::error::ContextError>,
    ) -> Self {
        let context = error.inner().context();

        // Use the first context that is a label
        // let label = context
        //     .find(|c| matches!(c, StrContext::Label(_)))
        //     .unwrap_or(&StrContext::Label("unknown"));

        // Use the Expected entries after the first label, until the next label
        let expected = context
            .skip_while(|c| matches!(c, StrContext::Label(_)))
            .take_while(|c| matches!(c, StrContext::Expected(_)))
            .filter_map(|c| match c {
                StrContext::Expected(e) => Some(e.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();

        let expected = expected.join(", ");

        let found = error
            .input()
            .get(error.offset())
            .map(|t| format!("{:?}", t.kind))
            .unwrap_or("EOF".into());

        let offset = if error.input().is_empty() {
            0
        } else {
            error
                .input()
                .get(error.offset())
                .map(|e| e.span.start)
                .unwrap_or_else(|| {
                    if error.offset() == 0 {
                        return 0;
                    }
                    // Unexpected EOF, so get the end of the last token's span
                    error
                        .input()
                        .get(error.offset() - 1)
                        .map(|e| e.span.end)
                        .unwrap_or(0)
                })
        };

        ParserError {
            message: format!("Expected {}, found {:?}", expected, found,),
            expected,
            found,
            offset,
        }
    }
}

type Tokens<'i> = TokenSlice<'i, Token>;

impl PartialEq<TokenKind> for Token {
    fn eq(&self, other: &TokenKind) -> bool {
        self.kind == *other
    }
}

impl winnow::stream::ContainsToken<&'_ Token> for TokenKind {
    #[inline(always)]
    fn contains_token(&self, token: &'_ Token) -> bool {
        *self == token.kind
    }
}

impl winnow::stream::ContainsToken<&'_ Token> for &'_ [TokenKind] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.contains(&token.kind)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<&'_ Token> for &'_ [TokenKind; LEN] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.contains(&token.kind)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<&'_ Token> for [TokenKind; LEN] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.contains(&token.kind)
    }
}

pub(crate) fn parse(input: &[Token]) -> Result<Program, ParserError> {
    let tokens = Tokens::new(input);
    let program = program.parse(tokens).map_err(ParserError::from_parse)?;
    Ok(program)
}

fn program(i: &mut Tokens<'_>) -> ModalResult<Program> {
    let function_declarations = repeat(
        0..,
        function_declaration
            .context(StrContext::Label("program"))
            .context(StrContext::Expected(StrContextValue::Description(
                "function declaration",
            ))),
    )
    .parse_next(i)?;
    Ok(Program {
        function_declarations,
    })
}

fn function_declaration(i: &mut Tokens<'_>) -> ModalResult<FunDecl> {
    literal(TokenKind::Keyword(Keyword::Int))
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    // After this point, any error is fatal

    cut_err(|i: &mut Tokens| {
        let name = identifier
            .context(StrContext::Label("function"))
            .context(StrContext::Expected(StrContextValue::Description(
                "identifier",
            )))
            .parse_next(i)?;

        let params = parameter_list.parse_next(i)?;

        let body = alt((block.map(Some), literal(TokenKind::Semicolon).map(|_| None)))
            .context(StrContext::Label("function"))
            .context(StrContext::Expected(StrContextValue::Description("block")))
            .parse_next(i)?;

        Ok(FunDecl { name, params, body })
    })
    .parse_next(i)
}

fn parameter_list(i: &mut Tokens<'_>) -> ModalResult<Vec<Identifier>> {
    literal(TokenKind::OpenParen)
        .context(StrContext::Label("parameter list"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("(")))
        .parse_next(i)?;

    let parameters = alt((
        literal(TokenKind::Keyword(Keyword::Void)).map(|_| vec![]),
        separated(
            // no parameters means an unspecified number of parameters, which we don't support yet
            1..,
            preceded(
                literal(TokenKind::Keyword(Keyword::Int))
                    .context(StrContext::Label("parameter list"))
                    .context(StrContext::Expected(StrContextValue::Description("int"))),
                identifier
                    .context(StrContext::Label("parameter list"))
                    .context(StrContext::Expected(StrContextValue::Description(
                        "identifier",
                    ))),
            ),
            literal(TokenKind::Comma)
                .context(StrContext::Label("parameter list"))
                .context(StrContext::Expected(StrContextValue::StringLiteral(","))),
        ),
        fail.context(StrContext::Label("parameter list"))
            .context(StrContext::Expected(StrContextValue::Description(
                "keyword",
            ))),
    ))
    .parse_next(i)?;

    literal(TokenKind::CloseParen)
        .context(StrContext::Label("paremeter list"))
        .context(StrContext::Expected(StrContextValue::StringLiteral(")")))
        .parse_next(i)?;

    Ok(parameters)
}

fn argument_list(i: &mut Tokens<'_>) -> ModalResult<Vec<Expression>> {
    literal(TokenKind::OpenParen)
        .context(StrContext::Label("argument list"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("(")))
        .parse_next(i)?;

    terminated(
        separated(
            0..,
            exp.context(StrContext::Label("argument list"))
                .context(StrContext::Expected(StrContextValue::Description(
                    "expression",
                ))),
            literal(TokenKind::Comma)
                .context(StrContext::Label("argument list"))
                .context(StrContext::Expected(StrContextValue::StringLiteral(","))),
        ),
        literal(TokenKind::CloseParen)
            .context(StrContext::Label("argument list"))
            .context(StrContext::Expected(StrContextValue::StringLiteral(")"))),
    )
    .parse_next(i)
}

fn block(i: &mut Tokens<'_>) -> ModalResult<Block> {
    literal(TokenKind::OpenBrace)
        .context(StrContext::Label("block"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("{")))
        .parse_next(i)?;

    let (items, _) = repeat_till(0.., cut_err(block_item), literal(TokenKind::CloseBrace))
        .context(StrContext::Label("block"))
        .context(StrContext::Expected(StrContextValue::Description(
            "block item",
        )))
        .parse_next(i)?;

    Ok(Block { items })
}

fn block_item(i: &mut Tokens<'_>) -> ModalResult<BlockItem> {
    let first = peek(any)
        .context(StrContext::Label("block_item"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword or statement",
        )))
        .parse_next(i)?;

    if let TokenKind::Keyword(Keyword::Int) = first.kind {
        let decl = declaration
            .context(StrContext::Label("block_item"))
            .context(StrContext::Expected(StrContextValue::Description(
                "declaration",
            )))
            .parse_next(i)?;
        Ok(BlockItem::D(decl))
    } else {
        let stmt = statement
            .context(StrContext::Label("block_item"))
            .context(StrContext::Expected(StrContextValue::Description(
                "statement",
            )))
            .parse_next(i)?;
        Ok(BlockItem::S(stmt))
    }
}

fn declaration(i: &mut Tokens<'_>) -> ModalResult<Declaration> {
    alt((
        variable_declaration.map(Declaration::VarDecl),
        function_declaration.map(Declaration::FunDecl),
    ))
    .context(StrContext::Label("declaration"))
    .context(StrContext::Expected(StrContextValue::Description(
        "declaration",
    )))
    .parse_next(i)
}

fn variable_declaration(i: &mut Tokens<'_>) -> ModalResult<VarDecl> {
    literal(TokenKind::Keyword(Keyword::Int))
        .context(StrContext::Label("variable declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    let name = identifier
        .context(StrContext::Label("variable declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "identifier",
        )))
        .parse_next(i)?;

    let next_token = peek(any)
        .context(StrContext::Label("variable declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "assignment or semicolon",
        )))
        .parse_next(i)?;

    let init = if next_token.kind == TokenKind::Assignment {
        take(1usize).parse_next(i)?;
        let exp = exp
            .context(StrContext::Label("variable declaration"))
            .context(StrContext::Expected(StrContextValue::Description(
                "expression",
            )))
            .parse_next(i)?;
        Some(exp)
    } else {
        None
    };

    literal(TokenKind::Semicolon)
        .context(StrContext::Label("variable declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;

    Ok(VarDecl { name, init })
}

fn statement(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    alt((labeled_statement, statement2)).parse_next(i)
}

fn labeled_statement(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    let label = identifier
        .context(StrContext::Label("labeled statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "identifier",
        )))
        .parse_next(i)?;

    literal(TokenKind::Colon)
        .context(StrContext::Label("labeled statement"))
        .context(StrContext::Expected(StrContextValue::Description(":")))
        .parse_next(i)?;

    let stmt = statement
        .context(StrContext::Label("labeled statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "statement",
        )))
        .parse_next(i)?;

    Ok(Statement::Labeled {
        label,
        statement: Box::new(stmt),
    })
}

fn statement2(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    dispatch! { peek(any);
        &Token { kind: TokenKind::Keyword(Keyword::Return), .. } => {
            statement_return
        },
        &Token { kind: TokenKind::Keyword(Keyword::If), .. } => {
            statement_if
        },
        &Token { kind: TokenKind::Keyword(Keyword::Goto), .. } => {
            statement_goto
        },
        &Token { kind: TokenKind::OpenBrace, .. } => {
            block.map(Statement::Compound)
        },
        &Token { kind: TokenKind::Keyword(Keyword::Break), .. } => {
            statement_break
        },
        &Token { kind: TokenKind::Keyword(Keyword::Continue), .. } => {
            statement_continue
        },
        &Token { kind: TokenKind::Keyword(Keyword::While), ..} => {
            statement_while
        },
        &Token { kind: TokenKind::Keyword(Keyword::Do), ..} => {
            statement_do_while
        },
        &Token { kind: TokenKind::Keyword(Keyword::For), ..} => {
            statement_for
        },
        &Token { kind: TokenKind::Semicolon, .. } => any.value(Statement::Null),
        _ => terminated(exp.map(Statement::Expression), literal(TokenKind::Semicolon)),
    }
    .context(StrContext::Label("statement"))
    .context(StrContext::Expected(StrContextValue::Description(
        "keyword",
    )))
    .context(StrContext::Expected(StrContextValue::Description(
        "expression",
    )))
    .context(StrContext::Expected(StrContextValue::Description(";")))
    .parse_next(i)
}

fn statement_return(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::Return))
        .context(StrContext::Label("return statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    let exp = exp
        .context(StrContext::Label("return statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "expression",
        )))
        .parse_next(i)?;
    literal(TokenKind::Semicolon)
        .context(StrContext::Label("return statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;
    Ok(Statement::Return(exp))
}

fn statement_if(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::If))
        .context(StrContext::Label("if statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    literal(TokenKind::OpenParen)
        .context(StrContext::Label("if statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "open parenthesis",
        )))
        .parse_next(i)?;
    let exp = exp
        .context(StrContext::Label("if statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "expression",
        )))
        .parse_next(i)?;
    literal(TokenKind::CloseParen)
        .context(StrContext::Label("if statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "close parenthesis",
        )))
        .parse_next(i)?;
    let then_stmt = statement
        .context(StrContext::Label("if statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "then statement",
        )))
        .parse_next(i)?;

    let next_token = peek(any)
        .context(StrContext::Label("if statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    let maybe_else_stmt = if next_token.kind == TokenKind::Keyword(Keyword::Else) {
        take(1usize).parse_next(i)?;
        let stmt = statement
            .context(StrContext::Label("if statement"))
            .context(StrContext::Expected(StrContextValue::Description(
                "else statement",
            )))
            .parse_next(i)?;
        Some(stmt)
    } else {
        None
    };

    Ok(Statement::If {
        condition: exp,
        then: Box::new(then_stmt),
        else_: maybe_else_stmt.map(Box::new),
    })
}

fn statement_goto(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::Goto))
        .context(StrContext::Label("goto statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    let label = identifier
        .context(StrContext::Label("goto statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "identifier",
        )))
        .parse_next(i)?;
    literal(TokenKind::Semicolon)
        .context(StrContext::Label("goto statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;
    Ok(Statement::Goto(label))
}

fn statement_break(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::Break))
        .context(StrContext::Label("break statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    literal(TokenKind::Semicolon)
        .context(StrContext::Label("break statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;

    Ok(Statement::Break(None))
}

fn statement_continue(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::Continue))
        .context(StrContext::Label("continue statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    literal(TokenKind::Semicolon)
        .context(StrContext::Label("continue statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;

    Ok(Statement::Continue(None))
}

fn statement_while(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::While))
        .context(StrContext::Label("while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    literal(TokenKind::OpenParen)
        .context(StrContext::Label("while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "open parenthesis",
        )))
        .parse_next(i)?;
    let condition = exp
        .context(StrContext::Label("while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "expression",
        )))
        .parse_next(i)?;
    literal(TokenKind::CloseParen)
        .context(StrContext::Label("while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "close parenthesis",
        )))
        .parse_next(i)?;
    let body = statement
        .context(StrContext::Label("while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "statement",
        )))
        .parse_next(i)?;
    Ok(Statement::While {
        condition,
        body: Box::new(body),
        loop_label: None,
    })
}

fn statement_do_while(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::Do))
        .context(StrContext::Label("do-while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    let body = statement
        .context(StrContext::Label("do-while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "statement",
        )))
        .parse_next(i)?;
    literal(TokenKind::Keyword(Keyword::While))
        .context(StrContext::Label("do-while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    literal(TokenKind::OpenParen)
        .context(StrContext::Label("do-while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "open parenthesis",
        )))
        .parse_next(i)?;
    let condition = exp
        .context(StrContext::Label("do-while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "expression",
        )))
        .parse_next(i)?;
    literal(TokenKind::CloseParen)
        .context(StrContext::Label("do-while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "close parenthesis",
        )))
        .parse_next(i)?;
    literal(TokenKind::Semicolon)
        .context(StrContext::Label("do-while statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;
    Ok(Statement::DoWhile {
        body: Box::new(body),
        condition,
        loop_label: None,
    })
}

fn statement_for(i: &mut Tokens<'_>) -> ModalResult<Statement> {
    literal(TokenKind::Keyword(Keyword::For))
        .context(StrContext::Label("for statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    literal(TokenKind::OpenParen)
        .context(StrContext::Label("for statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "open parenthesis",
        )))
        .parse_next(i)?;

    let init = for_init.parse_next(i)?;

    // semicolon consumed by for_init

    let condition =
        opt(exp
            .context(StrContext::Label("for statement"))
            .context(StrContext::Expected(StrContextValue::Description(
                "expression",
            ))))
        .parse_next(i)?;

    literal(TokenKind::Semicolon)
        .context(StrContext::Label("for statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "open parenthesis",
        )))
        .parse_next(i)?;

    let post = opt(exp
        .context(StrContext::Label("for statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "expression",
        ))))
    .parse_next(i)?;

    literal(TokenKind::CloseParen)
        .context(StrContext::Label("for statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "open parenthesis",
        )))
        .parse_next(i)?;

    let body = Box::new(
        statement
            .context(StrContext::Label("for statement"))
            .context(StrContext::Expected(StrContextValue::Description(
                "statement",
            )))
            .parse_next(i)?,
    );

    Ok(Statement::For {
        init,
        condition,
        post,
        body,
        loop_label: None,
    })
}

fn for_init(i: &mut Tokens<'_>) -> ModalResult<ForInit> {
    // FIXME: this might be wrong? Needs to handle the FunDecl case properly.
    // TODO: write a test to verify this works correctly.
    alt((
        declaration.try_map(|x| match x {
            Declaration::VarDecl(var_decl) => Ok(ForInit::InitDecl(var_decl)),
            _ => Err(ParserError {
                message: "Expected a variable declaration".to_string(),
                expected: "variable declaration".to_string(),
                found: format!("{:?}", x),
                offset: 0, // TODO how do we get the offset here?
            }),
        }),
        terminated(
            exp.map(Some).map(ForInit::InitExp),
            literal(TokenKind::Semicolon),
        ),
        literal(TokenKind::Semicolon).map(|_| ForInit::InitExp(None)),
    ))
    .parse_next(i)
}

/// Parses an expression with operator precedence, using Precedence Climbing.
fn exp_internal(i: &mut Tokens<'_>, min_prec: usize) -> ModalResult<Expression> {
    trace("exp_internal", |i: &mut Tokens<'_>| {
        let mut left = factor.parse_next(i)?;

        if i.is_empty() {
            return Ok(left);
        }
        let mut next_token = peek(any).parse_next(i)?;

        while next_token.is_binary_operator() && next_token.precedence() >= min_prec {
            if next_token.kind == TokenKind::Assignment {
                take(1usize).parse_next(i)?;
                let right = exp_internal(i, next_token.precedence())?;
                left = Expression::Assignment(left.into(), right.into());
            } else if next_token.kind == TokenKind::QuestionMark {
                let middle = conditional_middle.parse_next(i)?;
                let right = exp_internal(i, next_token.precedence())?;
                left = Expression::Conditional(left.into(), middle.into(), right.into());
            } else {
                let operator = binop.parse_next(i)?;
                let right = exp_internal(i, next_token.precedence() + 1)?;
                left = Expression::Binary(operator, left.into(), right.into());
            }

            if i.is_empty() {
                return Ok(left);
            }

            next_token = peek(any).parse_next(i)?;
        }
        Ok(left)
    })
    .parse_next(i)
}

fn exp(i: &mut Tokens<'_>) -> ModalResult<Expression> {
    exp_internal(i, 0)
}

fn factor(i: &mut Tokens<'_>) -> ModalResult<Expression> {
    trace("factor", |i: &mut _| {
        let next_token: &Token = peek(any).parse_next(i)?;
        let exp = match &next_token.kind {
            TokenKind::Constant(_) => Expression::Constant(int.parse_next(i)?),
            TokenKind::Identifier(identifier) => {
                take(1usize).parse_next(i)?;

                // Variable or function call?
                let next_token: &Token = peek(any).parse_next(i)?;
                match next_token.kind {
                    TokenKind::OpenParen => {
                        let arguments = argument_list.parse_next(i)?;
                        Expression::FunctionCall(identifier.clone(), arguments)
                    }
                    _ => Expression::Var(identifier.clone()),
                }
            }
            TokenKind::BitwiseComplement | TokenKind::Negation | TokenKind::LogicalNot => {
                let op = unop.parse_next(i)?;
                let inner_exp = factor.parse_next(i)?;
                Expression::Unary(op, Box::new(inner_exp))
            }
            TokenKind::OpenParen => {
                take(1usize).parse_next(i)?;
                let inner_exp = exp_internal(i, 0)?;
                literal(TokenKind::CloseParen).parse_next(i)?;
                inner_exp
            }
            _ => fail.context(StrContext::Label("factor")).parse_next(i)?,
        };

        Ok(exp)
    })
    .parse_next(i)
}

fn conditional_middle(i: &mut Tokens<'_>) -> ModalResult<Expression> {
    literal(TokenKind::QuestionMark)
        .context(StrContext::Label("conditional"))
        .context(StrContext::Expected(StrContextValue::Description("?")))
        .parse_next(i)?;
    let exp = exp
        .context(StrContext::Label("conditional"))
        .context(StrContext::Expected(StrContextValue::Description(
            "expression",
        )))
        .parse_next(i)?;
    literal(TokenKind::Colon)
        .context(StrContext::Label("conditional"))
        .context(StrContext::Expected(StrContextValue::Description(":")))
        .parse_next(i)?;
    Ok(exp)
}

fn identifier(i: &mut Tokens<'_>) -> ModalResult<Identifier> {
    let identifier = any
        .try_map(|t: &Token| {
            if let TokenKind::Identifier(ref id) = t.kind {
                Ok(id.clone())
            } else {
                Err(ParserError {
                    message: "Expected an identifier".to_string(),
                    expected: "identifier".to_string(),
                    found: format!("{:?}", t.kind),
                    offset: t.span.start,
                })
            }
        })
        .context(StrContext::Label("identifier"))
        .parse_next(i)?;
    Ok(identifier)
}

fn int(i: &mut Tokens<'_>) -> ModalResult<usize> {
    let constant = any
        .try_map(|t: &Token| {
            if let TokenKind::Constant(c) = t.kind {
                Ok(c)
            } else {
                Err(ParserError {
                    message: "Expected a constant".to_string(),
                    expected: "constant".to_string(),
                    found: format!("{:?}", t.kind),
                    offset: t.span.start,
                })
            }
        })
        .context(StrContext::Label("int"))
        .parse_next(i)?;
    Ok(constant)
}

fn unop(i: &mut Tokens<'_>) -> ModalResult<UnaryOperator> {
    let op = any
        .try_map(|t: &Token| match t.kind {
            TokenKind::BitwiseComplement => Ok(UnaryOperator::Complement),
            TokenKind::Negation => Ok(UnaryOperator::Negate),
            TokenKind::LogicalNot => Ok(UnaryOperator::Not),
            _ => Err(ParserError {
                message: "Expected a unary operator".to_string(),
                expected: "unary operator".to_string(),
                found: format!("{:?}", t.kind),
                offset: t.span.start,
            }),
        })
        .context(StrContext::Label("unop"))
        .parse_next(i)?;
    Ok(op)
}

fn binop(i: &mut Tokens<'_>) -> ModalResult<BinaryOperator> {
    let op = any
        .try_map(|t: &Token| match t.kind {
            TokenKind::Add => Ok(BinaryOperator::Add),
            TokenKind::Negation => Ok(BinaryOperator::Subtract),
            TokenKind::Multiply => Ok(BinaryOperator::Multiply),
            TokenKind::Divide => Ok(BinaryOperator::Divide),
            TokenKind::Remainder => Ok(BinaryOperator::Remainder),
            TokenKind::BitwiseAnd => Ok(BinaryOperator::BitAnd),
            TokenKind::BitwiseOr => Ok(BinaryOperator::BitOr),
            TokenKind::BitwiseXor => Ok(BinaryOperator::BitXor),
            TokenKind::BitwiseShiftLeft => Ok(BinaryOperator::ShiftLeft),
            TokenKind::BitwiseShiftRight => Ok(BinaryOperator::ShiftRight),
            TokenKind::LogicalAnd => Ok(BinaryOperator::And),
            TokenKind::LogicalOr => Ok(BinaryOperator::Or),
            TokenKind::Equal => Ok(BinaryOperator::Equal),
            TokenKind::NotEqual => Ok(BinaryOperator::NotEqual),
            TokenKind::LessThan => Ok(BinaryOperator::LessThan),
            TokenKind::GreaterThan => Ok(BinaryOperator::GreaterThan),
            TokenKind::LessThanOrEqual => Ok(BinaryOperator::LessOrEqual),
            TokenKind::GreaterThanOrEqual => Ok(BinaryOperator::GreaterOrEqual),
            _ => Err(ParserError {
                message: "Expected a binary operator".to_string(),
                expected: "binary operator".to_string(),
                found: format!("{:?}", t.kind),
                offset: t.span.start,
            }),
        })
        .context(StrContext::Label("binop"))
        .parse_next(i)?;
    Ok(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_c::*;
    use crate::lexer::lex;
    use assert_matches::assert_matches;
    use winnow::error::ParseError;
    use winnow::Parser;

    fn parse_program(input: &str) -> Program {
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let result = program.parse(tokens);
        result.expect("program should parse")
    }

    fn parse_function_declaration(input: &str) -> FunDecl {
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let result = function_declaration.parse(tokens);
        result.expect("function should parse")
    }

    fn parse_statement(input: &str) -> Statement {
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let result = statement.parse(tokens);
        result.expect("statement should parse")
    }

    fn parse_expression(input: &str) -> Expression {
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let result = exp.parse(tokens);
        result.expect("expression should parse")
    }

    #[test]
    fn test_bring_up() {
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        let _ = parse_program(input);
    }

    #[test]
    fn test_expression_constant() {
        let input = r#"2"#;
        let ast = parse_expression(input);
        assert_eq!(ast, Expression::Constant(2));
    }

    #[test]
    fn test_expression_bitwise_complement() {
        let input = r#"~2"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Unary(UnaryOperator::Complement, Box::new(Expression::Constant(2)))
        );
    }

    #[test]
    fn test_expression_negation() {
        let input = r#"-3"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Unary(UnaryOperator::Negate, Box::new(Expression::Constant(3)))
        );
    }

    #[test]
    fn test_expression_complement_of_negation() {
        let input = r#"~(-4)"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Unary(
                UnaryOperator::Complement,
                Box::new(Expression::Unary(
                    UnaryOperator::Negate,
                    Box::new(Expression::Constant(4))
                ))
            )
        );
    }

    #[test]
    fn test_expression_error() {
        let input = r#";"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let error = exp.parse(tokens).unwrap_err();
        assert_matches!(error, ParseError { .. });
    }

    #[test]
    fn test_expression_precedence_1() {
        // Figure 3-1
        let input = r#"1 + (2 * 3)"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Binary(
                BinaryOperator::Add,
                Box::new(Expression::Constant(1)),
                Box::new(Expression::Binary(
                    BinaryOperator::Multiply,
                    Box::new(Expression::Constant(2)),
                    Box::new(Expression::Constant(3))
                ))
            )
        );
    }

    #[test]
    fn test_expression_precedence_2() {
        // Figure 3-2
        let input = r#"(1 + 2) * 3"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Binary(
                BinaryOperator::Multiply,
                Box::new(Expression::Binary(
                    BinaryOperator::Add,
                    Box::new(Expression::Constant(1)),
                    Box::new(Expression::Constant(2))
                )),
                Box::new(Expression::Constant(3)),
            )
        );
    }
    #[test]
    fn test_expression_precedence_3() {
        // Page 55: 1 * 2 - 3 * (4 + 5)
        let input = r#"1 * 2 - 3 * (4 + 5)"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Binary(
                BinaryOperator::Subtract,
                Box::new(Expression::Binary(
                    BinaryOperator::Multiply,
                    Box::new(Expression::Constant(1)),
                    Box::new(Expression::Constant(2))
                )),
                Box::new(Expression::Binary(
                    BinaryOperator::Multiply,
                    Box::new(Expression::Constant(3)),
                    Box::new(Expression::Binary(
                        BinaryOperator::Add,
                        Box::new(Expression::Constant(4)),
                        Box::new(Expression::Constant(5))
                    ))
                ))
            )
        );
    }

    #[test]
    fn test_expression_bitwise_binary_operators() {
        // 80 >> 2 | 1 ^ 5 & 7 << 1
        // Equivalent to:
        // (80 >> 2) | (1 ^ (5 & (7 << 1)))
        let input = r#"80 >> 2 | 1 ^ 5 & 7 << 1"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Binary(
                BinaryOperator::BitOr,
                Box::new(Expression::Binary(
                    BinaryOperator::ShiftRight,
                    Box::new(Expression::Constant(80)),
                    Box::new(Expression::Constant(2))
                )),
                Box::new(Expression::Binary(
                    BinaryOperator::BitXor,
                    Box::new(Expression::Constant(1)),
                    Box::new(Expression::Binary(
                        BinaryOperator::BitAnd,
                        Box::new(Expression::Constant(5)),
                        Box::new(Expression::Binary(
                            BinaryOperator::ShiftLeft,
                            Box::new(Expression::Constant(7)),
                            Box::new(Expression::Constant(1))
                        ))
                    ))
                ))
            )
        );
    }

    #[test]
    fn test_expression_logical_binary_operators() {
        // !(80 && 2 == 1 || 5 <= 7 && 2 > 1)
        // Equivalent to:
        // !((80 && (2 == 1)) || ((5 <= 7) && (2 > 1)))
        let input = r#"!(80 && 2 == 1 || 5 <= 7 && 2 > 1)"#;
        let ast = parse_expression(input);
        assert_eq!(
            ast,
            Expression::Unary(
                UnaryOperator::Not,
                Box::new(Expression::Binary(
                    BinaryOperator::Or,
                    Box::new(Expression::Binary(
                        BinaryOperator::And,
                        Box::new(Expression::Constant(80)),
                        Box::new(Expression::Binary(
                            BinaryOperator::Equal,
                            Box::new(Expression::Constant(2)),
                            Box::new(Expression::Constant(1))
                        )),
                    )),
                    Box::new(Expression::Binary(
                        BinaryOperator::And,
                        Box::new(Expression::Binary(
                            BinaryOperator::LessOrEqual,
                            Box::new(Expression::Constant(5)),
                            Box::new(Expression::Constant(7))
                        )),
                        Box::new(Expression::Binary(
                            BinaryOperator::GreaterThan,
                            Box::new(Expression::Constant(2)),
                            Box::new(Expression::Constant(1))
                        ))
                    ))
                ))
            )
        )
    }

    #[test]
    fn test_declarations_and_assignments() {
        // Listing 5-3
        let input = r#"
        int main(void) {
            int a;
            a = 2;
            return a * 2;
        }"#;
        let ast = parse_program(input);

        assert_eq!(
            ast,
            Program {
                function_declarations: vec![FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: Some(Block {
                        items: vec![
                            BlockItem::D(Declaration::VarDecl(VarDecl {
                                name: "a".into(),
                                init: None,
                            })),
                            BlockItem::S(Statement::Expression(Expression::Assignment(
                                Box::new(Expression::Var("a".into())),
                                Box::new(Expression::Constant(2))
                            ))),
                            BlockItem::S(Statement::Return(Expression::Binary(
                                BinaryOperator::Multiply,
                                Box::new(Expression::Var("a".into())),
                                Box::new(Expression::Constant(2))
                            )))
                        ]
                    }),
                }]
            }
        )
    }

    #[test]
    fn test_nested_if() {
        // Listing 6-5
        let input = r#"
        if (a > 100)
            return 0;
        else if (a > 50)
            return 1;
        else
            return 2;"#;
        let ast = parse_statement(input);

        assert_eq!(
            ast,
            Statement::If {
                condition: Expression::Binary(
                    BinaryOperator::GreaterThan,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(100))
                ),
                then: Box::new(Statement::Return(Expression::Constant(0))),
                else_: Some(Box::new(Statement::If {
                    condition: Expression::Binary(
                        BinaryOperator::GreaterThan,
                        Box::new(Expression::Var("a".into())),
                        Box::new(Expression::Constant(50))
                    ),
                    then: Box::new(Statement::Return(Expression::Constant(1))),
                    else_: Some(Box::new(Statement::Return(Expression::Constant(2))))
                }))
            }
        )
    }

    #[test]
    fn test_parse_ternary_1() {
        // Page 122
        let input = r#"a = 1 ? 2 : 3"#;
        let ast = parse_expression(input);

        assert_eq!(
            ast,
            Expression::Assignment(
                Box::new(Expression::Var("a".into())),
                Box::new(Expression::Conditional(
                    Box::new(Expression::Constant(1)),
                    Box::new(Expression::Constant(2)),
                    Box::new(Expression::Constant(3))
                ))
            )
        )
    }

    #[test]
    fn test_parse_ternary_2() {
        // Page 122
        let input = r#"a || b ? 2 : 3"#;
        let ast = parse_expression(input);

        // Parses as (a || b) ? 2 : 3
        assert_eq!(
            ast,
            Expression::Conditional(
                Box::new(Expression::Binary(
                    BinaryOperator::Or,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Var("b".into()))
                )),
                Box::new(Expression::Constant(2)),
                Box::new(Expression::Constant(3))
            )
        )
    }

    #[test]
    fn test_parse_ternary_3() {
        // Page 122
        let input = r#"1 ? 2 : 3 || 4"#;
        let ast = parse_expression(input);

        // Parses as 1 ? 2 : (3 || 4)
        assert_eq!(
            ast,
            Expression::Conditional(
                Box::new(Expression::Constant(1)),
                Box::new(Expression::Constant(2)),
                Box::new(Expression::Binary(
                    BinaryOperator::Or,
                    Box::new(Expression::Constant(3)),
                    Box::new(Expression::Constant(4))
                )),
            )
        )
    }

    #[test]
    fn test_parse_ternary_4() {
        // Page 122
        let input = r#"x ? x = 1 : 2"#;
        let ast = parse_expression(input);

        // Parses as x ? (x = 1) : 2
        assert_eq!(
            ast,
            Expression::Conditional(
                Box::new(Expression::Var("x".into())),
                Box::new(Expression::Assignment(
                    Box::new(Expression::Var("x".into())),
                    Box::new(Expression::Constant(1))
                )),
                Box::new(Expression::Constant(2)),
            )
        )
    }

    #[test]
    fn test_parse_ternary_5() {
        // Page 123
        let input = r#"a ? b ? 1 : 2 : 3"#;
        let ast = parse_expression(input);

        // Parses as a ? (b ? 1 : 2) : 3
        assert_eq!(
            ast,
            Expression::Conditional(
                Box::new(Expression::Var("a".into())),
                Box::new(Expression::Conditional(
                    Box::new(Expression::Var("b".into())),
                    Box::new(Expression::Constant(1)),
                    Box::new(Expression::Constant(2)),
                )),
                Box::new(Expression::Constant(3)),
            )
        )
    }

    #[test]
    fn test_parse_ternary_6() {
        // Page 123
        let input = r#"a ? 1 : b ? 2 : 3"#;
        let ast = parse_expression(input);

        // Parses as a ? 1 : (b ? 2 : 3)  [right associative]
        assert_eq!(
            ast,
            Expression::Conditional(
                Box::new(Expression::Var("a".into())),
                Box::new(Expression::Constant(1)),
                Box::new(Expression::Conditional(
                    Box::new(Expression::Var("b".into())),
                    Box::new(Expression::Constant(2)),
                    Box::new(Expression::Constant(3)),
                )),
            )
        )
    }

    #[test]
    fn test_goto() {
        let input = r#"
        int main(void) {
            goto label1;
            label0: ;
            label1: return 0;
        }
        "#;
        let ast = parse_function_declaration(input);

        assert_eq!(
            ast,
            FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        BlockItem::S(Statement::Goto("label1".into())),
                        BlockItem::S(Statement::Labeled {
                            label: "label0".into(),
                            statement: Box::new(Statement::Null)
                        }),
                        BlockItem::S(Statement::Labeled {
                            label: "label1".into(),
                            statement: Box::new(Statement::Return(Expression::Constant(0)))
                        })
                    ]
                })
            }
        );
    }

    #[test]
    fn test_goto_nested_label() {
        // book-tests/tests/chapter_6/valid/extra_credit/goto_nested_label.c
        let input = r#"
        int main(void) {
            goto labelB;
            labelA:
                labelB:
                    return 5;
            return 0;
        }
        "#;
        let ast = parse_function_declaration(input);

        assert_eq!(
            ast,
            FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        BlockItem::S(Statement::Goto("labelB".into())),
                        BlockItem::S(Statement::Labeled {
                            label: "labelA".into(),
                            statement: Box::new(Statement::Labeled {
                                label: "labelB".into(),
                                statement: Box::new(Statement::Return(Expression::Constant(5)))
                            })
                        }),
                        BlockItem::S(Statement::Return(Expression::Constant(0))),
                    ]
                })
            }
        );
    }

    #[test]
    fn test_goto_backwards() {
        // book-tests/tests/chapter_6/valid/extra_credit/goto_backwards.c
        let input = r#"
        int main(void) {
            if (0)
            label:
                return 5;
            goto label;
            return 0;
        }
        "#;
        let ast = parse_function_declaration(input);

        assert_eq!(
            ast,
            FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        BlockItem::S(Statement::If {
                            condition: Expression::Constant(0),
                            then: Box::new(Statement::Labeled {
                                label: "label".into(),
                                statement: Box::new(Statement::Return(Expression::Constant(5)))
                            }),
                            else_: None,
                        }),
                        BlockItem::S(Statement::Goto("label".into())),
                        BlockItem::S(Statement::Return(Expression::Constant(0))),
                    ]
                })
            }
        );
    }

    #[test]
    fn test_compound_statements() {
        // Listing 7-4: Multiple nested scopes
        let input = r#"
        int main(void) {
            int x = 1;
            {
                int x = 2;
                if (x > 1) {
                    x = 3;
                    int x = 4;
                }
                return x;
            }
            return x;
        }
        "#;
        let ast = parse_function_declaration(input);

        assert_eq!(
            ast,
            FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        BlockItem::D(Declaration::VarDecl(VarDecl {
                            name: "x".into(),
                            init: Some(Expression::Constant(1)),
                        })),
                        BlockItem::S(Statement::Compound(Block {
                            items: vec![
                                BlockItem::D(Declaration::VarDecl(VarDecl {
                                    name: "x".into(),
                                    init: Some(Expression::Constant(2)),
                                })),
                                BlockItem::S(Statement::If {
                                    condition: Expression::Binary(
                                        BinaryOperator::GreaterThan,
                                        Box::new(Expression::Var("x".into())),
                                        Box::new(Expression::Constant(1))
                                    ),
                                    then: Box::new(Statement::Compound(Block {
                                        items: vec![
                                            BlockItem::S(Statement::Expression(
                                                Expression::Assignment(
                                                    Box::new(Expression::Var("x".into())),
                                                    Box::new(Expression::Constant(3))
                                                )
                                            )),
                                            BlockItem::D(Declaration::VarDecl(VarDecl {
                                                name: "x".into(),
                                                init: Some(Expression::Constant(4)),
                                            }))
                                        ]
                                    })),
                                    else_: None,
                                }),
                                BlockItem::S(Statement::Return(Expression::Var("x".into())))
                            ]
                        })),
                        BlockItem::S(Statement::Return(Expression::Var("x".into())))
                    ]
                })
            }
        );
    }

    #[test]
    fn test_while_statement() {
        let input = r#"
        int main(void) {
            while (1) {
                if (0)
                    continue;
                else
                    break;
            }
        }
        "#;
        let ast = parse_function_declaration(input);

        assert_eq!(
            ast,
            FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![BlockItem::S(Statement::While {
                        condition: Expression::Constant(1),
                        body: Box::new(Statement::Compound(Block {
                            items: vec![BlockItem::S(Statement::If {
                                condition: Expression::Constant(0),
                                then: Box::new(Statement::Continue(None)),
                                else_: Some(Box::new(Statement::Break(None))),
                            })]
                        })),
                        loop_label: None,
                    })]
                })
            }
        );
    }

    #[test]
    fn test_do_while_statement() {
        let input = r#"
        int main(void) {
            do {
                if (0)
                    continue;
                else
                    break;
            } while (1);
        }
        "#;
        let ast = parse_function_declaration(input);

        assert_eq!(
            ast,
            FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![BlockItem::S(Statement::DoWhile {
                        body: Box::new(Statement::Compound(Block {
                            items: vec![BlockItem::S(Statement::If {
                                condition: Expression::Constant(0),
                                then: Box::new(Statement::Continue(None)),
                                else_: Some(Box::new(Statement::Break(None))),
                            })]
                        })),
                        condition: Expression::Constant(1),
                        loop_label: None,
                    })]
                })
            }
        );
    }

    #[test]
    fn test_for_statement() {
        let input = r#"
        int main(void) {
            for (int i = 0; i < 10; i = i + 1) {
                if (i > 5)
                    break;
            }
        }
        "#;
        let ast = parse_function_declaration(input);

        assert_eq!(
            ast,
            FunDecl {
                name: "main".into(),
                params: vec![],
                body: Some(Block {
                    items: vec![BlockItem::S(Statement::For {
                        init: ForInit::InitDecl(VarDecl {
                            name: "i".into(),
                            init: Some(Expression::Constant(0)),
                        }),
                        condition: Some(Expression::Binary(
                            BinaryOperator::LessThan,
                            Box::new(Expression::Var("i".into())),
                            Box::new(Expression::Constant(10))
                        )),
                        post: Some(Expression::Assignment(
                            Box::new(Expression::Var("i".into())),
                            Box::new(Expression::Binary(
                                BinaryOperator::Add,
                                Box::new(Expression::Var("i".into())),
                                Box::new(Expression::Constant(1))
                            ))
                        )),
                        body: Box::new(Statement::Compound(Block {
                            items: vec![BlockItem::S(Statement::If {
                                condition: Expression::Binary(
                                    BinaryOperator::GreaterThan,
                                    Box::new(Expression::Var("i".into())),
                                    Box::new(Expression::Constant(5))
                                ),
                                then: Box::new(Statement::Break(None)),
                                else_: None,
                            })]
                        })),
                        loop_label: None,
                    })]
                })
            }
        );
    }

    #[test]
    fn test_for_statement_no_init() {
        let input = r#"
            for (; 0; 1)
                ;
        "#;
        let ast = parse_statement(input);

        assert_eq!(
            ast,
            Statement::For {
                init: ForInit::InitExp(None),
                condition: Some(Expression::Constant(0)),
                post: Some(Expression::Constant(1)),
                body: Box::new(Statement::Null),
                loop_label: None,
            }
        );
    }

    #[test]
    fn test_for_statement_init_exp() {
        let input = r#"
            for (2; 0; 1)
                ;
        "#;
        let ast = parse_statement(input);

        assert_eq!(
            ast,
            Statement::For {
                init: ForInit::InitExp(Some(Expression::Constant(2))),
                condition: Some(Expression::Constant(0)),
                post: Some(Expression::Constant(1)),
                body: Box::new(Statement::Null),
                loop_label: None,
            }
        );
    }

    #[test]
    fn test_for_statement_no_condition() {
        let input = r#"
            for (1; ; 1)
                ;
        "#;
        let ast = parse_statement(input);

        assert_eq!(
            ast,
            Statement::For {
                init: ForInit::InitExp(Some(Expression::Constant(1))),
                condition: None,
                post: Some(Expression::Constant(1)),
                body: Box::new(Statement::Null),
                loop_label: None,
            }
        );
    }

    #[test]
    fn test_for_statement_no_post() {
        let input = r#"
            for (1; 2; )
                ;
        "#;
        let ast = parse_statement(input);

        assert_eq!(
            ast,
            Statement::For {
                init: ForInit::InitExp(Some(Expression::Constant(1))),
                condition: Some(Expression::Constant(2)),
                post: None,
                body: Box::new(Statement::Null),
                loop_label: None,
            }
        );
    }

    #[test]
    fn test_for_statement_no_nothing() {
        let input = r#"
            for (;;)
                ;
        "#;
        let ast = parse_statement(input);

        assert_eq!(
            ast,
            Statement::For {
                init: ForInit::InitExp(None),
                condition: None,
                post: None,
                body: Box::new(Statement::Null),
                loop_label: None,
            }
        );
    }

    #[test]
    fn test_parameter_list() {
        fn parse(input: &str) -> Vec<Identifier> {
            let tokens = lex(input).unwrap();
            let tokens = Tokens::new(&tokens);
            let result = parameter_list.parse(tokens);
            result.expect("program should parse")
        }

        assert!(parse("(void)").is_empty());
        assert_eq!(parse("(int x)"), vec!["x".to_string()]);
        assert_eq!(
            parse("(int x, int y)"),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_argument_list() {
        fn parse(input: &str) -> Vec<Expression> {
            let tokens = lex(input).unwrap();
            let tokens = Tokens::new(&tokens);
            let result = argument_list.parse(tokens);
            result.expect("program should parse")
        }

        assert!(parse("()").is_empty());
        assert_eq!(parse("(x)"), vec![Expression::Var("x".to_string())]);
        assert_eq!(
            parse("(1, y + 2)"),
            vec![
                Expression::Constant(1),
                Expression::Binary(
                    BinaryOperator::Add,
                    Box::new(Expression::Var("y".to_string())),
                    Box::new(Expression::Constant(2))
                )
            ]
        );
    }

    #[test]
    fn test_function_declaration() {
        let input = r#"
            int main(void);
        "#;
        let ast = parse_program(input);

        assert_eq!(
            ast,
            Program {
                function_declarations: vec![FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: None
                }]
            }
        )
    }

    #[test]
    fn test_function_definitions_multiple() {
        let input = r#"
            int main(void) { return 1; }
            int foo(int x) { return x; }
            int bar(int x, int y) { return x + y; }
        "#;
        let ast = parse_program(input);

        assert_eq!(
            ast,
            Program {
                function_declarations: vec![
                    FunDecl {
                        name: "main".into(),
                        params: vec![],
                        body: Some(Block {
                            items: vec![BlockItem::S(Statement::Return(Expression::Constant(1)))]
                        }),
                    },
                    FunDecl {
                        name: "foo".into(),
                        params: vec!["x".into()],
                        body: Some(Block {
                            items: vec![BlockItem::S(Statement::Return(Expression::Var(
                                "x".into()
                            )))]
                        }),
                    },
                    FunDecl {
                        name: "bar".into(),
                        params: vec!["x".into(), "y".into()],
                        body: Some(Block {
                            items: vec![BlockItem::S(Statement::Return(Expression::Binary(
                                BinaryOperator::Add,
                                Box::new(Expression::Var("x".into())),
                                Box::new(Expression::Var("y".into()))
                            )))]
                        }),
                    }
                ]
            }
        )
    }

    #[test]
    fn test_function_declaration_in_function() {
        let input = r#"
            int main(void) {
                int x;
                int helper(int y);
                return helper(42);
            }
        "#;
        let ast = parse_program(input);

        assert_eq!(
            ast,
            Program {
                function_declarations: vec![FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: Some(Block {
                        items: vec![
                            BlockItem::D(Declaration::VarDecl(VarDecl {
                                name: "x".into(),
                                init: None,
                            })),
                            BlockItem::D(Declaration::FunDecl(FunDecl {
                                name: "helper".into(),
                                params: vec!["y".into()],
                                body: None,
                            })),
                            BlockItem::S(Statement::Return(Expression::FunctionCall(
                                "helper".into(),
                                vec![Expression::Constant(42)],
                            ))),
                        ]
                    }),
                }]
            }
        )
    }
}
