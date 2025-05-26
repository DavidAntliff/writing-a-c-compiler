//! Parser for the C language
//!
//! Grammar:
//!   <program> ::= <function>
//!   <function> ::= "int" <identifier> "(" "void" ")" "{" { <block-item> } "}"
//!   <block-item> ::= <statement> | <declaration>
//!   <declaration> ::= "int" <identifier> [ "=" <exp> ] ";"
//!   <statement> ::= "return" <exp> ";" | <exp> ";" | ";"
//!   <exp> := <factor> | <exp> <binop> <exp>
//!   <primary-exp> := <int> | <identifier> | "(" <exp> ")"
//!   <postfix-exp> := <primary-exp> { "++" | "--" }
//!   <factor> ::= <unop> <factor> | <postfix-exp>
//!   <unop> ::= "-" | "~" | "!" | "++" | "--"
//!   <binop> ::= "-" | "+" | "-" | "*" | "/" | "%"
//!             | "&" | "|" | "^" | "<<" | ">>"
//!             | "&&" | "||" | "==" | "!="
//!             | "<" | "<=" | ">" | ">="
//!   <identifier> ::= ? An identifier token ?
//!   <int> ::= ? A constant token ?
//!

use crate::ast_c::{
    BinaryOperator, BlockItem, Declaration, Expression, Function, Program, Statement, UnaryOperator,
};
use crate::lexer::{Keyword, Token, TokenKind};
use thiserror::Error;
use winnow::combinator::{fail, peek, repeat_till, terminated, trace};
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
        self.iter().any(|t| *t == token.kind)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<&'_ Token> for &'_ [TokenKind; LEN] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.iter().any(|t| *t == token.kind)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<&'_ Token> for [TokenKind; LEN] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.iter().any(|t| *t == token.kind)
    }
}

pub(crate) fn parse(input: &[Token]) -> Result<Program, ParserError> {
    let tokens = Tokens::new(input);
    let program = program.parse(tokens).map_err(ParserError::from_parse)?;
    Ok(program)
}

fn program(i: &mut Tokens<'_>) -> winnow::Result<Program> {
    let function = function
        .context(StrContext::Label("program"))
        .context(StrContext::Expected(StrContextValue::Description(
            "function",
        )))
        .parse_next(i)?;
    Ok(Program { function })
}

fn function(i: &mut Tokens<'_>) -> winnow::Result<Function> {
    literal(TokenKind::Keyword(Keyword::Int))
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    let name = identifier
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "identifier",
        )))
        .parse_next(i)?;

    literal(TokenKind::OpenParen)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("(")))
        .parse_next(i)?;

    literal(TokenKind::Keyword(Keyword::Void))
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    literal(TokenKind::CloseParen)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral(")")))
        .parse_next(i)?;

    literal(TokenKind::OpenBrace)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("{")))
        .parse_next(i)?;

    let (body, _) = repeat_till(0.., block_item, literal(TokenKind::CloseBrace))
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "block_item",
        )))
        .parse_next(i)?;

    Ok(Function { name, body })
}

fn block_item(i: &mut Tokens<'_>) -> winnow::Result<BlockItem> {
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

fn declaration(i: &mut Tokens<'_>) -> winnow::Result<Declaration> {
    literal(TokenKind::Keyword(Keyword::Int))
        .context(StrContext::Label("declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;

    let name = identifier
        .context(StrContext::Label("declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "identifier",
        )))
        .parse_next(i)?;

    let next_token = peek(any)
        .context(StrContext::Label("declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "assignment or semicolon",
        )))
        .parse_next(i)?;

    let init = if next_token.kind == TokenKind::Assignment {
        take(1usize).parse_next(i)?;
        let exp = exp
            .context(StrContext::Label("declaration"))
            .context(StrContext::Expected(StrContextValue::Description(
                "expression",
            )))
            .parse_next(i)?;
        Some(exp)
    } else {
        None
    };

    literal(TokenKind::Semicolon)
        .context(StrContext::Label("declaration"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;

    Ok(Declaration { name, init })
}

fn statement(i: &mut Tokens<'_>) -> winnow::Result<Statement> {
    dispatch! { peek(any);
        &Token { kind: TokenKind::Keyword(Keyword::Return), .. } => statement_return,
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

fn statement_return(i: &mut Tokens<'_>) -> winnow::Result<Statement> {
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

/// Parses an expression with operator precedence, using Precedence Climbing.
fn exp_internal(i: &mut Tokens<'_>, min_prec: usize) -> winnow::Result<Expression> {
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
                left = Expression::Assignment(left.into(), right.into())
            } else {
                let operator = binop.parse_next(i)?;
                let right = exp_internal(i, next_token.precedence() + 1)?;
                left = Expression::Binary(operator, left.into(), right.into());

                if i.is_empty() {
                    return Ok(left);
                }
            }
            next_token = peek(any).parse_next(i)?;
        }
        Ok(left)
    })
    .parse_next(i)
}

fn exp(i: &mut Tokens<'_>) -> winnow::Result<Expression> {
    exp_internal(i, 0)
}

fn primary_exp(i: &mut Tokens<'_>) -> winnow::Result<Expression> {
    trace("factor", |i: &mut _| {
        let next_token: &Token = peek(any).parse_next(i)?;
        let exp = match next_token.kind {
            TokenKind::Constant(_) => Expression::Constant(int.parse_next(i)?),
            TokenKind::Identifier(_) => Expression::Var(identifier.parse_next(i)?),
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

fn postfix_exp(i: &mut Tokens<'_>) -> winnow::Result<Expression> {
    trace("postfix_exp", |i: &mut _| {
        let primary = primary_exp.parse_next(i)?;
        postfix_helper(i, primary)
    })
    .parse_next(i)
}

fn postfix_helper(i: &mut Tokens<'_>, mut expr: Expression) -> winnow::Result<Expression> {
    loop {
        let next_token: &Token = peek(any).parse_next(i)?;
        match next_token.kind {
            TokenKind::Increment => {
                take(1usize).parse_next(i)?;
                expr = Expression::PostFixIncrement(Box::new(expr));
            }
            TokenKind::Decrement => {
                take(1usize).parse_next(i)?;
                expr = Expression::PostFixDecrement(Box::new(expr));
            }
            _ => return Ok(expr),
        }
    }
}

fn factor(i: &mut Tokens<'_>) -> winnow::Result<Expression> {
    trace("factor", |i: &mut _| {
        let next_token: &Token = peek(any).parse_next(i)?;
        let exp = match next_token.kind {
            TokenKind::BitwiseComplement
            | TokenKind::Negation
            | TokenKind::LogicalNot
            | TokenKind::Increment
            | TokenKind::Decrement => {
                let operator = unop.parse_next(i)?;
                let inner_exp = factor.parse_next(i)?;
                Expression::Unary(operator, Box::new(inner_exp))
            }
            _ => fail.context(StrContext::Label("factor")).parse_next(i)?,
        };
        Ok(exp)
    })
    .parse_next(i)
}

fn identifier(i: &mut Tokens<'_>) -> winnow::Result<String> {
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

fn int(i: &mut Tokens<'_>) -> winnow::Result<usize> {
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

fn unop(i: &mut Tokens<'_>) -> winnow::Result<UnaryOperator> {
    let op = any
        .try_map(|t: &Token| match t.kind {
            TokenKind::BitwiseComplement => Ok(UnaryOperator::Complement),
            TokenKind::Negation => Ok(UnaryOperator::Negate),
            TokenKind::Increment => Ok(UnaryOperator::Increment),
            TokenKind::Decrement => Ok(UnaryOperator::Decrement),
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

fn binop(i: &mut Tokens<'_>) -> winnow::Result<BinaryOperator> {
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
    use crate::ast_c::*;
    use crate::lexer::lex;
    use crate::parser::{exp, program, Tokens};
    use assert_matches::assert_matches;
    use winnow::error::ParseError;
    use winnow::Parser;

    #[test]
    fn test_bring_up() {
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        let tokens = lex(input).unwrap();
        //dbg!(&tokens);
        let mut tokens = Tokens::new(&tokens);
        let _program = crate::parser::program.parse_next(&mut tokens);
        //dbg!(&program);
    }

    #[test]
    fn test_expression_constant() {
        let input = r#"2"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(expression, Expression::Constant(2));
    }

    #[test]
    fn test_expression_bitwise_complement() {
        let input = r#"~2"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(
            expression,
            Expression::Unary(UnaryOperator::Complement, Box::new(Expression::Constant(2)))
        );
    }

    #[test]
    fn test_expression_negation() {
        let input = r#"-3"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(
            expression,
            Expression::Unary(UnaryOperator::Negate, Box::new(Expression::Constant(3)))
        );
    }

    #[test]
    fn test_expression_complement_of_negation() {
        let input = r#"~(-4)"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        eprintln!("***");
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(
            expression,
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
        crate::tests::init_logger();

        let input = r#"1 + (2 * 3)"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).expect("should parse");
        assert_eq!(
            expression,
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
        crate::tests::init_logger();

        let input = r#"(1 + 2) * 3"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).expect("should parse");
        assert_eq!(
            expression,
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
        crate::tests::init_logger();

        let input = r#"1 * 2 - 3 * (4 + 5)"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).expect("should parse");
        assert_eq!(
            expression,
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
        crate::tests::init_logger();

        let input = r#"80 >> 2 | 1 ^ 5 & 7 << 1"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).expect("should parse");
        assert_eq!(
            expression,
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
        crate::tests::init_logger();

        let input = r#"!(80 && 2 == 1 || 5 <= 7 && 2 > 1)"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).expect("should parse");
        assert_eq!(
            expression,
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
        crate::tests::init_logger();

        let input = r#"
        int main(void) {
            int a;
            a = 2;
            return a * 2;
        }"#;
        let tokens = lex(input).unwrap();
        let mut tokens = Tokens::new(&tokens);
        let program = program.parse_next(&mut tokens).expect("should parse");

        assert_eq!(
            program,
            Program {
                function: Function {
                    name: "main".into(),
                    body: vec![
                        BlockItem::D(Declaration {
                            name: "a".into(),
                            init: None,
                        }),
                        BlockItem::S(Statement::Expression(Expression::Assignment(
                            Box::new(Expression::Var("a".into())),
                            Box::new(Expression::Constant(2))
                        ))),
                        BlockItem::S(Statement::Return(Expression::Binary(
                            BinaryOperator::Multiply,
                            Box::new(Expression::Var("a".into())),
                            Box::new(Expression::Constant(2))
                        )))
                    ],
                }
            }
        )
    }
}
